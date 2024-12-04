import random
import numpy as np
import torch as ch  
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist    
from modules.helper import off_diagonal, GatherLayer, gather_center

################################
##### Loss definitions #####
################################

class SimCLRLoss(nn.Module):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.0, 
    weight decay of 1e-6 and a temperature of 0.15.
    """
    def __init__(self, cfg, batch_size, world_size, gpu):
        super(SimCLRLoss, self).__init__()
        temperature = cfg.pretrain.simclr.temperature

        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size).to(gpu)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = ch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.world_size

        if self.world_size > 1:
            z_i = ch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = ch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = ch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = ch.matmul(features, features.T)/ self.temperature

        sim_i_j = ch.diag(sim, batch_size * self.world_size)
        sim_j_i = ch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = ch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        logits = ch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = ch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (- logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim, num_entropy


class BarlowTwinsLoss(nn.Module):
    def __init__(self, cfg, bn, batch_size, world_size):
        lambd = cfg.pretrain.barlow.lambd

        super(BarlowTwinsLoss, self).__init__()
        self.cfg = cfg
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
        if self.cfg.pretrain.training.distributed:
            ch.distributed.all_reduce(c)

        on_diag = ch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class ByolLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        momentum_teacher = cfg.pretrain.byol.momentum_teacher

        self.momentum_teacher = momentum_teacher

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.chunk(2)
        teacher_out = teacher_output.detach().chunk(2)

        student_out_1, student_out_2 = student_out
        student_out_1 = F.normalize(student_out_1, dim=-1, p=2)
        student_out_2 = F.normalize(student_out_2, dim=-1, p=2)
        teacher_out_1, teacher_out_2 = teacher_out
        teacher_out_1 = F.normalize(teacher_out_1, dim=-1, p=2)
        teacher_out_2 = F.normalize(teacher_out_2, dim=-1, p=2)
        loss_1 = 2 - 2 * (student_out_1 * teacher_out_2.detach()).sum(dim=-1)
        loss_2 = 2 - 2 * (student_out_2 * teacher_out_1.detach()).sum(dim=-1)
        return (loss_1 + loss_2).mean()
    

class DINOLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        out_dim = cfg.pretrain.dino.out_dim
        ncrops = cfg.pretrain.dino.local_crops_n + cfg.pretrain.dino.global_crops_n
        warmup_teacher_temp = cfg.pretrain.dino.warmup_teacher_temp
        teacher_temp = cfg.pretrain.dino.teacher_temp
        warmup_teacher_temp_epochs = cfg.pretrain.dino.warmup_teacher_temp_epochs
        nepochs = cfg.pretrain.training.epochs

        self.student_temp = cfg.pretrain.dino.student_temp
        self.center_momentum = cfg.pretrain.dino.center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", ch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = ch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @ch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = ch.sum(teacher_output, dim=0, keepdim=True)
        if self.cfg.pretrain.training.distributed:
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else: 
            batch_center = batch_center / len(teacher_output)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)