# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/libffcv/ffcv-imagenet to support SSL

import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import time
import json
import uuid
import ffcv
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from modules.net import SSLNetwork
from modules.utils import LARS, cosine_scheduler, learning_schedule
from modules.losses import SimCLRLoss,  BarlowTwinsLoss, ByolLoss, MixupLoss
from modules.datasets import create_val_loader, create_train_loader_ssl, create_train_loader_supervised


################################
##### Some Miscs functions #####
################################

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    path = "/checkpoint/"
    if Path(path).is_dir():
        p = Path(f"{path}{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file



class LinearsProbes(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        mlp_coeff = cfg.pretrain.model.mlp_coeff
        print("NUM CLASSES", num_classes)
        mlp_spec = f"{model.representation_size}-{model.mlp}"
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]


################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    def __init__(self, cfg, gpu, ngpus_per_node, world_size, dist_url):
        self.cfg = cfg
        distributed = cfg.pretrain.training.distributed
        batch_size = cfg.pretrain.training.batch_size
        label_smoothing = cfg.pretrain.training.label_smoothing
        loss = cfg.pretrain.training.loss
        train_probes_only = cfg.pretrain.training.train_probes_only
        epochs = cfg.pretrain.training.epochs
        mixup = cfg.pretrain.training.mixup
        train_dataset = cfg.data.train_dataset
        val_dataset = cfg.data.val_dataset


        # self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.device = f'cuda:{gpu}'
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.seed = 50 + self.rank
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()

        self.start_epoch = 0
        
        # Create DataLoader
        self.train_dataset = train_dataset
        self.index_labels = 1
        self.train_loader, self.decoder, self.decoder2 = create_train_loader_ssl(cfg, self.gpu, train_dataset)
        self.num_train_exemples = self.train_loader.indices.shape[0]
        self.num_classes = 1000
        self.val_loader = create_val_loader(cfg, self.gpu, val_dataset)
        print("NUM TRAINING EXEMPLES:", self.num_train_exemples)
        
        # Create SSL model
        self.model, self.scaler = self.create_model_and_scaler(cfg)
        if distributed:
            self.model_module = self.model_module
        else:
            self.model_module = self.model

        
        self.num_features = self.model_module.num_features
        self.n_layers_proj = len(self.model_module.projector) + 1

        
        print("N layers in proj:", self.n_layers_proj)
        self.initialize_logger(cfg)
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer(cfg)
        
        # Create lineares probes
        self.loss = nn.CrossEntropyLoss()
        self.probes = LinearsProbes(cfg, self.model_module, num_classes=self.num_classes)
        self.probes = self.probes.to(memory_format=ch.channels_last)
        self.probes = self.probes.to(self.gpu)
        if cfg.pretrain.training.distributed:
            self.probes = ch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu])
        self.optimizer_probes = ch.optim.AdamW(self.probes.parameters(), lr=1e-4)
        # Load models if checkpoints
        self.load_checkpoint(cfg)
        # Define SSL loss
        self.do_ssl_training = False if train_probes_only else True
        self.teacher_student = False
        self.supervised_loss = False
        self.loss_name = loss
        if loss == "simclr":
            self.ssl_loss = SimCLRLoss(cfg, batch_size, world_size, self.gpu).to(self.gpu)
        elif loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(cfg, self.model_module.bn, batch_size, world_size)
        elif loss == "byol":
            self.ssl_loss = ByolLoss(cfg)
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler()
            self.teacher.module.load_state_dict(self.model_module.state_dict())
            self.momentum_schedule = cosine_scheduler(self.ssl_loss.momentum_teacher, 1, epochs, len(self.train_loader))
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif loss == "supervised":
            self.supervised_loss = True
        else:
            print("Loss not available")
            exit(1)

    # resolution tools

    def get_resolution(self, cfg, epoch): 
        min_res = cfg.data.resolution.min_res
        max_res = cfg.data.resolution.max_res
        end_ramp = cfg.data.resolution.end_ramp
        start_ramp = cfg.data.resolution.start_ramp
        
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res


    def get_dataloader(self, cfg):
        use_ssl = cfg.pretrain.training.use_ssl
        train_dataset = cfg.data.train_dataset
        if use_ssl:
            train_loader, self.decoder, self.decoder2 = create_train_loader_ssl(cfg, self.gpu, train_dataset)
            return train_loader, self.create_val_loader()
        else:
            train_loader, self.decoder, self.decoder2 = create_train_loader_supervised(cfg, self.gpu, train_dataset)
            return train_loader, self.create_val_loader()

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def create_optimizer(self, cfg):
        momentum = cfg.pretrain.training.momentum
        optimizer = cfg.pretrain.training.optimizer
        weight_decay = cfg.pretrain.training.weight_decay
        label_smoothing = cfg.pretrain.training.label_smoothing

        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optim_name = optimizer


    def train(self, cfg):
        epochs = cfg.pretrain.training.epochs
        log_level = cfg.pretrain.logging.log_level

        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_exemples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            res = self.get_resolution(cfg, epoch)
            self.res = res
            self.decoder.output_size = (res, res)
            self.decoder2.output_size = (res, res)
            train_loss, stats = self.train_loop(cfg, epoch)
            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }
                self.log(dict(stats,  **extra_dict))
            self.eval_and_log(stats, extra_dict)
            # Run checkpointing
            self.checkpoint(cfg, epoch + 1)
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, stats, extra_dict={}):
        stats = self.val_loop()
        self.log(dict(stats, **extra_dict))
        return stats

    def create_model_and_scaler(self, cfg):
        loss = cfg.pretrain.training.loss

        scaler = ch.amp.GradScaler(ch.cuda.current_device())
        # scaler = GradScaler()
        model = SSLNetwork(cfg)
        if loss == "supervised":
            model.fc = nn.Linear(model.num_features, self.num_classes)
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if cfg.pretrain.training.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        # model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    def load_checkpoint(self, cfg):
        train_probes_only = cfg.pretrain.training.train_probes_only

        if (self.log_folder / "model.pth").is_file():
            if self.rank == 0:
                print("resuming from checkpoint")
            ckpt = ch.load(self.log_folder / "model.pth", map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if not train_probes_only:
                self.probes.load_state_dict(ckpt["probes"])
                self.optimizer_probes.load_state_dict(ckpt["optimizer_probes"])
            else:
                self.start_epoch = 0

    def checkpoint(self, cfg, epoch):
        checkpoint_freq = cfg.pretrain.logging.checkpoint_freq
        train_probes_only = cfg.pretrain.training.train_probes_only

        if self.rank != 0 or epoch % checkpoint_freq != 0:
            return
        if train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"probes.pth"
        else:
            state = dict(
                epoch=epoch, 
                model=self.model.state_dict(), 
                optimizer=self.optimizer.state_dict(),
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"model.pth"
        ch.save(state, self.log_folder / save_name)

    def train_loop(self, cfg, epoch):
        """
        Main training loop for SSL training with VicReg criterion.
        """
        log_level = cfg.pretrain.logging.log_level
        base_lr = cfg.pretrain.training.base_lr
        end_lr_ratio = cfg.pretrain.training.end_lr_ratio
        # mixup = cfg.pretrain.training.mixup
        model = self.model
        model.train()
        losses = []

        iterator = tqdm(self.train_loader)
        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):

            # Get lr
            lr = learning_schedule(
                global_step=ix,
                batch_size=self.batch_size * self.world_size,
                base_lr=base_lr,
                end_lr_ratio=end_lr_ratio,
                total_steps=self.max_steps,
                warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
            )
            for g in self.optimizer.param_groups:
                 g["lr"] = lr

            # Get data
            images_big_0 = loaders[0]
            labels_big = loaders[1]
            batch_size = loaders[1].size(0)
            images_big_1 = loaders[2]
            images_big = ch.cat((images_big_0, images_big_1), dim=0)

            # SSL Training
            if self.do_ssl_training:
                self.optimizer.zero_grad(set_to_none=True)
                with ch.amp.autocast(self.device):
                    if self.teacher_student:
                        with ch.no_grad():
                            teacher_output, _ = self.teacher(images_big)
                            teacher_output = teacher_output.view(2, batch_size, -1)
                        embedding_big, _ = model(images_big, predictor=True)
                    elif self.supervised_loss:
                        embedding_big, _ = model(images_big_0.repeat(2,1,1,1))
                    else:
                        # Compute embedding in bigger crops
                        embedding_big, _ = model(images_big)
                    
                    # Compute SSL Loss
                    if self.teacher_student:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        loss_train = self.ssl_loss(embedding_big, teacher_output)
                    elif self.supervised_loss:
                        output_classif_projector = self.model_module.fc(embedding_big)
                        loss_train = self.classif_loss(output_classif_projector, labels_big.repeat(2))
                    else:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        if "simclr" in self.loss_name:
                            loss_num, loss_denum = self.ssl_loss(embedding_big[0], embedding_big[1])
                            loss_train = loss_num + loss_denum
                        else:
                            loss_train = self.ssl_loss(embedding_big[0], embedding_big[1])
                            
                    self.scaler.scale(loss_train).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss_train = ch.tensor(0.)
            if self.teacher_student:
                m = self.momentum_schedule[ix]  # momentum parameter
                for param_q, param_k in zip(model_module.parameters(), self.teacher.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Online linear probes training
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_probes.zero_grad(set_to_none=True)
            # Compute embeddings vectors
            with ch.no_grad():
                with ch.amp.autocast(self.device):
                    _, list_representation = model(images_big_0)
            # Train probes
            with ch.amp.autocast(self.device):
                # Real value classification
                list_outputs = self.probes(list_representation)
                loss_classif = 0.
                for l in range(len(list_outputs)):
                    # Compute classif loss
                    current_loss = self.loss(list_outputs[l], labels_big)
                    loss_classif += current_loss
                    self.train_meters['loss_classif_layer'+str(l)](current_loss.detach())
                    for k in ['top_1_layer'+str(l), 'top_5_layer'+str(l)]:
                        self.train_meters[k](list_outputs[l].detach(), labels_big)
            self.scaler.scale(loss_classif).backward()
            self.scaler.step(self.optimizer_probes)
            self.scaler.update()

            # Logging
            if log_level > 0:
                self.train_meters['loss'](loss_train.detach())
                losses.append(loss_train.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images_big.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']
                    names += ['loss_c']
                    values += [f'{loss_classif.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats

    def val_loop(self):
        model = self.model
        model.eval()
        with ch.no_grad():
            with ch.amp.autocast(self.device): 
                for images, target in tqdm(self.val_loader):
                    _, list_representation = model(images)
                    list_outputs = self.probes(list_representation)
                    loss_classif = 0.
                    for l in range(len(list_outputs)):
                        # Compute classif loss
                        current_loss = self.loss(list_outputs[l], target)
                        loss_classif += current_loss
                        self.val_meters['loss_classif_val_layer'+str(l)](current_loss.detach())
                        for k in ['top_1_val_layer'+str(l), 'top_5_val_layer'+str(l)]:
                            self.val_meters[k](list_outputs[l].detach(), target)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    def initialize_logger(self, cfg):
        folder = cfg.pretrain.logging.folder

        self.train_meters = {
            'loss': torchmetrics.MeanMetric().to(self.gpu),
            'time': torchmetrics.MeanMetric().to(self.gpu),
        }

        for l in range(self.n_layers_proj):
            self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
            self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, ).to(self.gpu)
            self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, ).to(self.gpu)

        self.val_meters = {}
        for l in range(self.n_layers_proj):
            self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
            self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, ).to(self.gpu)
            self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, ).to(self.gpu)

        if self.gpu == 0:
            if Path(folder + 'final_weights.pt').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            os.makedirs(folder, exist_ok=True)
            with open(folder / 'params.json', 'w+') as json_file:
                json.dump(cfg_dict, json_file, indent=4)
            # os.makedirs(self.log_folder, exist_ok=True)
            # params = {
            #     '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            # }

            # with open(folder / 'params.json', 'w+') as handle:
            #     json.dump(params, handle)
        self.log_folder = Path(folder)

    def log(self, content):
        train_probes_only = self.cfg.pretrain.training.train_probes_only
        print(f'=> Log: {content}')
        if self.rank != 0: return
        cur_time = time.time()
        name_file = 'log_probes' if train_probes_only else 'log'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    def launch_from_args(cls, cfg):
        distributed = cfg.pretrain.training.distributed
        port = str(cfg.distributed.port)
        world_size = cfg.distributed.world_size
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=( cfg, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0, cfg)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        # if args[1] is not None:
        #     set_current_config(args[1])
        # make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, cfg, ngpus_per_node=1, world_size=1, dist_url=None):
        distributed = cfg.pretrain.training.distributed
        eval_only = cfg.pretrain.training.eval_only
        

        trainer = cls(cfg=cfg, gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train(cfg)

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    parser.add_argument("folder", type=str)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config


def run_submitit(cfg):
    folder = cfg.pretrain.logging.folder
    ngpus = cfg.distributed.ngpus
    nodes = cfg.distributed.nodes
    timeout = cfg.distributed.timeout
    partition = cfg.distributed.partition
    comment = cfg.distributed.comment
    port = cfg.distributed.port

    Path(folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    # Cluster specifics: To update accordingly to your cluster
    kwargs = {}
    kwargs['slurm_comment'] = comment
    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="ffcv-ssl")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(cfg, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {folder}")

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def print_cfg(cfg):
    from prettytable import PrettyTable

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = flatten_dict(cfg_dict)
    table = PrettyTable()
    table.field_names = ["Key", "Value"]

    for key, value in flat_cfg.items():
        table.add_row([key, value])

    print(table)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    print_cfg(cfg)
    use_submitit = cfg.distributed.use_submitit
    if use_submitit:
        run_submitit(cfg)
    else:
        ImageNetTrainer.launch_from_args(cfg)

if __name__ == "__main__":
    # config = make_config()
    main()
