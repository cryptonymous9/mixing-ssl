import os
import sys
import json
import numpy as np
import glob
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, ImageFolder

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

import argparse

# python image_to_ffcv.py --data_dir="../in100/train/" --write_path="../data/in100-ffcv/train.beton" 
# pythoh image_to_ffcv.py --data_dir="../in100/train/" --diffusion=True --diffusion_dir="" --write_path="../data/in-g8-ffcv/train.beton"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../in100/train/", type=str)
parser.add_argument("--write_path", default="./", type=str, required=True)
parser.add_argument("--class_to_idx_path", default=None, type=str)
parser.add_argument("--idx_to_class_path", default=None, type=str)
parser.add_argument("--jpeg_quality", default=90, type=int)
parser.add_argument("--max_resolution", default=256, type=int)
parser.add_argument("--subset", default=-1, help="""No. of Images to convert (-1 for all)""", type=int)
parser.add_argument("--num_workers", default=16, type=int)
parser.add_argument("--chunk_size", default=400, type=int)
parser.add_argument("--write_mode", choices=["raw", "smart", "jpg"], default='smart', type=str)
parser.add_argument("--diffusion", type=bool, default=False)
parser.add_argument("--diffusion_dir", default='../in100-g/train/', type=str)

args = parser.parse_args()



class DualImageDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        """
        Args:
            root_dir1 (string): Directory with all the images from source 1.
            root_dir2 (string): Directory with all the images from source 2.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        
        self.classes = sorted(os.listdir(root_dir1))  # Assumes both directories have the same class subfolders
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # Assumes corresponding images in both directories have the same filename
        self.image_paths1 = glob.glob(os.path.join(root_dir1, '*/*'))
        self.image_paths2 = [path.replace(root_dir1, root_dir2) for path in self.image_paths1]

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):

        img_path1 = self.image_paths1[idx]
        img_path2 = self.image_paths2[idx]
                
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        # if len(img1_shape)!=3:
            # img1 = np.stack((np.array(img1),)*3, axis=-1)
            # img1 = np.reshape(img1, (img1_shape[0], img1_shape[1],1))
            # print(img_path1, np.shape(img1))
            # print(img_path2, np.shape(img2))
            # sys.exit() 
        # if img1_shape[-1]==4:
        #     print(img_path1)
        # if img2_shape[-1]==4:
        #     print(img_path1)
            
        label = self.class_to_idx[os.path.basename(os.path.dirname(img_path1))]
        # print(np.shape(img1), np.shape(img2), label, type(label))
        
        return img1, img2, int(label)
    

class ClassDataset(Dataset):
    def __init__(self, root, class_to_idx_path, idx_to_class_path=None, transform=None):
        """
        Args:
            root (string): Root directory of the dataset.
            class_to_idx_path (string): Path to the JSON file containing class to index mapping.
            transform (callable, optional): A function/transform to apply to the images.
        """
        
        # Load the class to index mapping
        with open(class_to_idx_path, 'r') as json_file:
            self.class_to_idx = json.load(json_file)
            
        if idx_to_class_path:
            with open(idx_to_class_path, 'r') as json_file:
                idx_to_class = json.load(json_file)
                reversed_idx_to_class = {v: k for k, v in idx_to_class.items()}

            classes = [idx_to_class[d] for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        else:
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        dirs = [d for d in classes if d in self.class_to_idx.keys()]

        self.images = []
        self.labels = []

        for d in dirs:
            label = self.class_to_idx[d]
            if idx_to_class_path:
                path = os.path.join(root, str(reversed_idx_to_class[d]))
            else:   
                path = os.path.join(root, d)
            images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPEG')]
            self.images.extend(images)
            self.labels.extend([label] * len(images))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        # Load image
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        # Apply transformation to the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    

def main(args):
    if args.diffusion:
        print(f"Data loading from '{args.data_dir}' & '{args.diffusion_dir}' directory")
        dataset = DualImageDataset(root_dir1=args.data_dir, root_dir2=args.diffusion_dir)
        print(f"Configuration: \n JPEG quality:{args.jpeg_quality}\t Resolution:{args.max_resolution}\t Mode:{args.write_mode}")
        if args.subset >0: dataset = Subset(dataset, range(args.subset))

        writer = DatasetWriter(args.write_path, {
            'image1':RGBImageField(write_mode=args.write_mode,
                                max_resolution=args.max_resolution,
                                jpeg_quality=args.jpeg_quality),
            'image2':RGBImageField(write_mode=args.write_mode,
                                max_resolution=args.max_resolution,
                                jpeg_quality=args.jpeg_quality),
            'label': IntField(),
        }, num_workers=args.num_workers)

        writer.from_indexed_dataset(dataset, chunksize=args.chunk_size)
        print(f"Both datasets written to '{args.write_path}' directory")

    else:
        
        if args.class_to_idx_path:
            print(f"read class_to_idx from {args.class_to_idx_path}")
            dataset = ClassDataset(root=args.data_dir,
                                   class_to_idx_path=args.class_to_idx_path,
                                   idx_to_class_path=args.idx_to_class_path)
        else:
            dataset = ImageFolder(root=args.data_dir)
        
        print(f"Configuration: \n JPEG quality:{args.jpeg_quality}\t Resolution:{args.max_resolution}\t Mode:{args.write_mode}")
        print(f"Data loading from '{args.data_dir}' directory!")
        
        if args.subset >0: dataset = Subset(dataset, range(args.subset))

        writer = DatasetWriter(args.write_path, {
            'image':RGBImageField(write_mode=args.write_mode,
                                max_resolution=args.max_resolution,
                                jpeg_quality=args.jpeg_quality),
            'label': IntField(),
        }, num_workers=args.num_workers)

        writer.from_indexed_dataset(dataset, chunksize=args.chunk_size)
        print(f"Data written to '{args.write_path}' directory")

if __name__ == '__main__':
    main(args)