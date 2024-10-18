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



IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

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
    def __init__(self, cfg, model, num_classes, mlp_coeff):
        super().__init__()
        mlp_coeff = cfg.pretrain.model.mlp_coeff
        print("NUM CLASSES", num_classes)
        mlp_spec = f"{model.module.representation_size}-{model.module.mlp}"
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]


################################in100-ffcv  in-g8_10per
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


        ch.cuda.set_device(gpu)
        self.gpu = gpu
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
        self.train_loader = self.get_dataloader(cfg)
        self.num_train_exemples = self.train_loader.indices.shape[0]
        self.num_classes = 100
        self.val_loader = self.create_val_loader(cfg, val_dataset)
        print("NUM TRAINING EXEMPLES:", self.num_train_exemples)
        # Create SSL model
        self.model, self.scaler = self.create_model_and_scaler(cfg)
        self.num_features = self.model.module.num_features
        self.n_layers_proj = len(self.model.module.projector) + 1
        print("N layers in proj:", self.n_layers_proj)
        self.initialize_logger(cfg)
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer(cfg)
        # Create lineares probes
        self.loss = nn.CrossEntropyLoss()
        self.probes = LinearsProbes(cfg, self.model, num_classes=self.num_classes)
        self.probes = self.probes.to(memory_format=ch.channels_last)
        self.probes = self.probes.to(self.gpu)
        self.probes = ch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu])
        self.optimizer_probes = ch.optim.AdamW(self.probes.parameters(), lr=1e-4)
        # Load models if checkpoints
        self.load_checkpoint(cfg)
        # Define SSL loss
        self.do_ssl_training = False if train_probes_only else True
        self.teacher_student = False
        self.supervised_loss = False
        self.loss_name = loss
        if mixup:
            self.mixup_loss = MixupLoss(batch_size, world_size, self.gpu)
        
        if loss == "simclr":
            self.ssl_loss = SimCLRLoss(cfg, batch_size, world_size, self.gpu).to(self.gpu)
        elif loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(cfg, self.model.module.bn, batch_size, world_size)
        elif loss == "byol":
            self.ssl_loss = ByolLoss(cfg)
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler(cfg)
            self.teacher.module.load_state_dict(self.model.module.state_dict())
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

        # print("Train", train_dataset)
        if use_ssl:
            return self.create_train_loader_ssl(cfg, train_dataset) #, _
        else:
            return self.create_train_loader_supervised(cfg, train_dataset) #, None

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


    def create_train_loader_ssl(self, cfg, train_dataset):
        batch_size = cfg.pretrain.training.batch_size
        distributed = cfg.pretrain.training.distributed
        
        num_workers = cfg.data.num_workers
        in_memory = cfg.data.in_memory

        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        # print("Train", train_path)
        # print("bool", train_path.is_file())
        # cwd = os.getcwd()
        # print(cwd)
        # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # print(hydra_cfg['runtime']['output_dir'])
        
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image1': image_pipeline_big,
            'image2': image_pipeline_big2,
            'label': label_pipeline,
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        #custom_field_mapper={"image_0": "image"}

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,)
                        #custom_field_mapper=custom_field_mapper)


        return loader

    def create_train_loader_supervised(self, cfg, train_dataset):
        
        batch_size = cfg.pretrain.training.batch_size
        distributed = cfg.pretrain.training.distributed
        num_workers = cfg.data.num_workers
        in_memory = cfg.data.in_memory

        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image1': image_pipeline_big,
            'image2': image_pipeline_big2,
            'label': label_pipeline,
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        #custom_field_mapper={"image_0": "image"}

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,)
                        #custom_field_mapper=custom_field_mapper)


        return loader


    def create_val_loader(self, cfg, val_dataset):
        batch_size = cfg.pretrain.validation.batch_size
        resolution = cfg.pretrain.validation.resolution
        distributed = cfg.pretrain.training.distributed
        num_workers = cfg.data.num_workers

        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    def train(self, cfg):
        epochs = cfg.pretrain.training.epochs
        log_level = cfg.logging.log_level
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


    def create_model_and_scaler(self, cfg ):
        loss = cfg.pretrain.training.loss

        scaler = GradScaler()
        model = SSLNetwork()
        if loss == "supervised":
            model.fc = nn.Linear(model.num_features, self.num_classes)
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
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

        if self.rank != 0 or (epoch+1) % checkpoint_freq != 0:
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
            save_name = f"model_{epoch+1}.pth"
        ch.save(state, self.log_folder / save_name)


    def train_loop(self, cfg, epoch):
        """
        Main training loop for SSL training with VicReg criterion.
        """
        log_level = cfg.pretrain.logging.log_level
        base_lr = cfg.pretrain.training.base_lr
        end_lr_ratio = cfg.pretrain.training.end_lr_ratio
        mixup = cfg.pretrain.training.mixup

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
            labels_big = loaders[2]
            batch_size = loaders[2].size(0)
            images_big_1 = loaders[1]
            images_big = ch.cat((images_big_0, images_big_1), dim=0)

            if mixup:
                mixup_alpha = 1 
                lam = ch.FloatTensor([np.random.beta(mixup_alpha, mixup_alpha)]).cuda()
                images_mixup = lam*images_big_0 + (1-lam)*images_big_1
            # SSL Training
            if self.do_ssl_training:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    if self.teacher_student:  # byol
                        with ch.no_grad():
                            teacher_output, _ = self.teacher(images_big)
                            teacher_output = teacher_output.view(2, batch_size, -1)
                        embedding_big, _ = model(images_big, predictor=True)
                    elif self.supervised_loss:
                        embedding_big, _ = model(images_big_0.repeat(2,1,1,1))
                    else:
                        # Compute embedding in bigger crops
                        embedding_big, _ = model(images_big)
                        if mixup:
                            embedding_mixup, _  = model(images_mixup)
                        
                    # Compute SSL Loss
                    if self.teacher_student: # byol
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        loss_train = self.ssl_loss(embedding_big, teacher_output)
                    elif self.supervised_loss:
                        output_classif_projector = model.module.fc(embedding_big)
                        loss_train = self.classif_loss(output_classif_projector, labels_big.repeat(2))
                    else:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        if "simclr" in self.loss_name:
                            loss_num, loss_denum = self.ssl_loss(embedding_big[0], embedding_big[1])
                            loss_train = loss_num + loss_denum
                        else:
                            loss_train = self.ssl_loss(embedding_big[0], embedding_big[1])
                        if mixup:
                            mloss = self.mixup_loss(embedding_big[0], embedding_big[1], embedding_mixup, lam)
                            # print(loss_train) if self.gpu==0 else None
                            # print(mloss) if self.gpu==0 else None
                            loss_train+=0.3*mloss
                    self.scaler.scale(loss_train).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss_train = ch.tensor(0.)
            if self.teacher_student:
                m = self.momentum_schedule[ix]  # momentum parameter
                for param_q, param_k in zip(model.module.parameters(), self.teacher.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Online linear probes training
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_probes.zero_grad(set_to_none=True)
            # Compute embeddings vectors
            with ch.no_grad():
                with autocast():
                    _, list_representation = model(images_big_0)
            # Train probes
            with autocast():
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
            with autocast():
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

        from pathlib import Path
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        self.train_meters = {
            'loss': torchmetrics.MeanMetric().to(self.gpu),
            'time': torchmetrics.MeanMetric().to(self.gpu),
        }

        for l in range(self.n_layers_proj):
            self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
            self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes).to(self.gpu)
            self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes).to(self.gpu)

        self.val_meters = {}
        for l in range(self.n_layers_proj):
            self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
            self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes).to(self.gpu)
            self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes).to(self.gpu)

        if self.gpu == 0:
            if Path(folder + 'final_weights.pt').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)

    def log(self, content, train_probes_only):
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
        port = cfg.distributed.port
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
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, cfg, ngpus_per_node, world_size, dist_url))
            # ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True)

        else:
            dist_url = None
            cls.exec(0, cfg, dist_url)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        # if args[1] is not None:
        #     set_current_config(args[1])
        # make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, cfg, dist_url):
        ngpus_per_node = ch.cuda.device_count()
        distributed = cfg.pretrain.training.distributed
        world_size = cfg.distributed.world_size
        dist_url = None
        eval_only = cfg.pretrain.training.eval_only



        trainer = cls(cfg=cfg, gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train(cfg)

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, cfg, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = cfg
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
        ImageNetTrainer._exec_wrapper(gpu, self.cfg, self.num_gpus_per_node, world_size, dist_url)

# Running
# def make_config(quiet=False):
#     config = get_current_config()
#     parser = ArgumentParser(description='Fast SSL training')
#     # parser.add_argument("folder", type=str)
#     config.augment_argparse(parser)
#     config.collect_argparse_args(parser)
#     config.validate(mode='stderr')
#     if not quiet:
#         config.summary()
#     return config


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

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    use_submitit = cfg.distributed.use_submitit
    if use_submitit:
        run_submitit(cfg)
    else:
        ImageNetTrainer.launch_from_args(cfg)

if __name__ == "__main__":
    main()