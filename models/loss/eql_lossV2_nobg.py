import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial


class EQLv2NoBg(nn.Module):
    def __init__(self,
                 num_classes,
                 loss_weight=1.0,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def expand_label(self, pred, gt_classes):
        """
        pred: [N, C]
        gt_classes:
            foreground: 0 ~ C-1
            # no BG
            background: C  (or any value >= C) 
        """
        N, C = pred.shape
        target = pred.new_zeros(N, C)
        fg_mask = (gt_classes >= 0) & (gt_classes < C)
        inds = torch.arange(N, device=pred.device)[fg_mask]
        target[inds, gt_classes[fg_mask]] = 1
        return target

    def get_weight(self):
        neg_w = self.map_func(self.pos_neg)          # [C]
        pos_w = 1 + self.alpha * (1 - neg_w)         # [C]
        return pos_w, neg_w

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=0)        # [C]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)  # [C]

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def forward(self, cls_score, label):
        """
        cls_score: [N, C]
        label:
            foreground: 0 ~ C-1
            background: C
        """
        N, C = cls_score.size()
        assert C == self.num_classes

        target = self.expand_label(cls_score, label)   # [N, C]

        pos_w, neg_w = self.get_weight()               # [C], [C]
        weight = pos_w.view(1, C) * target + neg_w.view(1, C) * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, target, reduction='none'
        )
        cls_loss = torch.sum(cls_loss * weight) / N

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_activation(self, cls_score):
        return torch.sigmoid(cls_score)