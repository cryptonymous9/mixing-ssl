import random
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
        lambd = cfg.pretrain.barlowtwins.lambd

        super(BarlowTwinsLoss, self).__init__()
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
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
    

class MixupLoss(nn.Module):
    """
    Mixup Loss:
    """
    def __init__(self, batch_size, world_size, gpu):
        super(MixupLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = None
        self.world_size = world_size
        
    def forward(self, z_i, z_j, z_mixup, lam):
        batch_size = z_i.size(0)

        if self.world_size > 1:
            z_i = gather_center(z_i/0.4)
            z_j = gather_center(z_j/0.4)
        # z_i = (z_i)/0.04
        # z_j = (z_j)/0.04
        # z_mixup = z_mixup/0.04
        z_k = lam*z_i + (1-lam)*z_j
        # z_mixup = ch.cat(GatherLayer.apply(z_mixup), dim=0)
        z_mixup = gather_center(z_mixup/0.4)
        
        # print(z_i.size(), z_j.size(), z_mixup.size())
        q = F.softmax(z_mixup, dim=-1)
        loss = ch.sum(-q*F.log_softmax(z_k, dim=-1),dim=-1)
        loss = loss.mean()
        return loss
    

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)