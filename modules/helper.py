import os
import time
import json
import uuid
from uuid import uuid4
from typing import List
from pathlib import Path
import torch.distributed as dist

import ffcv
import torch as ch

################################
##### Some Miscs functions #####
################################

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_shared_folder(path="./checkpoint/") -> Path:
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

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    x_list = GatherLayer.apply(x.contiguous())
    return ch.cat(x_list, dim=0)

class GatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]