import numpy as np
from typing import List
from pathlib import Path
import torch as ch
import torchvision.transforms as transforms

import ffcv
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder




IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256



def create_val_loader(cfg, gpu, val_dataset):
    batch_size = cfg.pretrain.validation.batch_size
    resolution = cfg.pretrain.validation.resolution
    distributed = cfg.pretrain.training.distributed
    num_workers = cfg.data.num_workers

    this_device = f'cuda:{gpu}'
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


def create_train_loader_ssl_mixdiff(cfg, gpu, train_dataset):
    batch_size = cfg.pretrain.training.batch_size
    distributed = cfg.pretrain.training.distributed
    
    num_workers = cfg.data.num_workers
    in_memory = cfg.data.in_memory

    this_device = f'cuda:{gpu}'
    train_path = Path(train_dataset)
    # print("Train", train_path)
    # print("bool", train_path.is_file())
    # cwd = os.getcwd()
    # print(cwd)
    # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # print(hydra_cfg['runtime']['output_dir'])
    
    assert train_path.is_file()
    # First branch of augmentations
    decoder = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big: List[Operation] = [
        decoder,
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
    decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big2: List[Operation] = [
        decoder2,
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

    order = OrderOption.RANDOM 
    #custom_field_mapper={"image_0": "image"}

    # Create data loader
    loader = ffcv.Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines=pipelines,
                    distributed=bool(distributed))
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
    decoder = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(ch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    # Second branch of augmentations
    decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big2: List[Operation] = [
        decoder2,
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

    order = OrderOption.RANDOM
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

    return loader, decoder, decoder2

def create_train_loader_ssl(cfg, gpu, train_dataset):
    batch_size = cfg.pretrain.training.batch_size
    distributed = cfg.pretrain.training.distributed
    
    num_workers = cfg.data.num_workers
    in_memory = cfg.data.in_memory

    this_device = f'cuda:{gpu}'
    train_path = Path(train_dataset)
    assert train_path.is_file()
    # First branch of augmentations
    decoder = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big: List[Operation] = [
        decoder,
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
    decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
    image_pipeline_big2: List[Operation] = [
        decoder2,
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
        'image': image_pipeline_big,
        'label': label_pipeline,
        'image_0': image_pipeline_big2
    }

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    custom_field_mapper={"image_0": "image"}

    # Create data loader

    loader = ffcv.Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines=pipelines,
                    distributed=distributed,
                    custom_field_mapper=custom_field_mapper)
    
    return loader, decoder, decoder2