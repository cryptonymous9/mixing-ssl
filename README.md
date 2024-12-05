## MixDiff

> Note: This repository is currently a work in progress!

[PyTorch] Code for the paper - MixDiff: Mixing Natural and Synthetic Images for Robust Self-Supervised Representations, WACV, 2025.

`MixDiff` is a self-supervised learning (SSL) pre-training framework that leverages both real and synthetic images to enhance representation learning. Unlike traditional SSL methods that rely heavily on real images, MixDiff introduces a novel approach by incorporating a variant of Stable Diffusion to replace an augmented instance of a real image. This enables the model to learn cross real-synthetic representations effectively. Our experiments confirm that MixDiff not only improves performance but also reduces the dependency on large amounts of real data, making it an efficient and versatile framework for SSL.


<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1WODFQ4ODPxfP1cZXMa4YdcRo84R8edAS" width="600">
</br>
<em>Comparison of SimCLR performance on real, synthetic (Syn), and mixed real and synthetic images (MixDiff). The radar charts show normalized accuracy across 8 transfer learning datasets (left) and ImageNet-1K plus 6 distribution shift datasets (right), with values from 0.5 to 1.1. MixDiff enhances in-distribution and robustness performance and generalizes better.</em>
</div>
</br>

## Installation

```
conda create -y -n ffcv-ssl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda=11.7 numba -c pytorch -c nvidia -c conda-forge
conda activate ffcv-ssl
pip install -e .
```
Troubleshooting note: if the above commands result in a package conflict error, try running ``conda config --env --set channel_priority flexible`` in the environment and rerunning the installation command. For detailed installation instructions, please refer to the [FFCV library](https://github.com/libffcv/ffcv) and [FFCV-SSL library](https://github.com/facebookresearch/FFCV-SSL).

## Generating Synthetic Images

To generate images using Stable Diffusion or Versatile Diffusion models from an input dataset, please refer to the [generating_images](./generating_images) folder.
