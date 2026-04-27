# Import from third library
import torch

from torch.nn.modules.loss import _Loss
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# class BaseLoss(_Loss):
#     # do not use syntax like `super(xxx, self).__init__,
#     # which will cause infinited recursion while using class decorator`
#     def __init__(self,
#                  name='base',
#                  reduction='none',
#                  loss_weight=1.0):
#         r"""
#         Arguments:
#             - name (:obj:`str`): name of the loss function
#             - reduction (:obj:`str`): reduction type, choice of mean, none, sum
#             - loss_weight (:obj:`float`): loss weight
#         """
#         _Loss.__init__(self, reduction=reduction)
#         self.loss_weight = loss_weight
#         self.name = name

#     def __call__(self, input, target, reduction_override=None, normalizer_override=None, **kwargs):
#         r"""
#         Arguments:
#             - input (:obj:`Tensor`)
#             - reduction (:obj:`Tensor`)
#             - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
#             defined in __init__ function

#             - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
#         """
#         reduction = reduction_override if reduction_override else self.reduction
#         assert (normalizer_override is None or reduction == 'mean'), \
#             f'normalizer is not allowed when reduction is {reduction}'
#         loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)
#         return loss * self.loss_weight

#     def forward(self, input, target, reduction, normalizer=None, **kwargs):
#         raise NotImplementedError


# class GeneralizedCrossEntropyLoss(BaseLoss):
#     def __init__(self,
#                  name='generalized_cross_entropy_loss',
#                  reduction='none',
#                  loss_weight=1.0,
#                  activation_type='softmax',
#                  ignore_index=-1,):
#         BaseLoss.__init__(self,
#                           name=name,
#                           reduction=reduction,
#                           loss_weight=loss_weight)
#         self.activation_type = activation_type
#         self.ignore_index = ignore_index


# @LOSSES_REGISTRY.register('equalized_focal_loss')
class EqualizedFocalLoss(nn.Module):
    def __init__(self,
                 name='equalized_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-2,
                 num_classes=8,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 warmup_epochs=0,
                 num_decoder_layers=6,
                 eps=1e-8
                 ):
        super().__init__()

        self.loss_weight = loss_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.scale_factor = scale_factor
        self.eps = eps
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.epoch = 0

        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # cfg for efl loss
        self.scale_factor = scale_factor

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.num_decoder_layers = num_decoder_layers
        self.cur_collect_idx = 0

        self.register_buffer('tmp_pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('tmp_neg_grad', torch.zeros(self.num_classes))



        # logger.info(f"build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma}, \
                    # scale_factor: {scale_factor}")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _expand_target(self, target_classes):
        """
        target_classes: [B, Q]
            foreground: 0 ~ C-1
        return:
            target: [B, Q, C]
            valid_mask: [B, Q]
        """
        B, Q = target_classes.shape
        target = torch.zeros(B, Q, self.num_classes, device=target_classes.device, dtype=torch.float32)

        valid_mask = (target_classes != self.ignore_index)
        fg_mask = (target_classes >= 0) & (target_classes < self.num_classes)

        if fg_mask.any():
            b_idx, q_idx = torch.where(fg_mask)
            cls_idx = target_classes[b_idx, q_idx]
            target[b_idx, q_idx, cls_idx] = 1.0

        return target, valid_mask


    def forward(self, pred_logits, target_classes, reduction=None, normalizer=None):

        """
        pred_logits:   [B, Q, C]
        target_classes:[B, Q], foreground: 0..C-1, background: -1
        """
        B, Q, C = pred_logits.shape
        assert C == self.num_classes

        target, valid_mask = self._expand_target(target_classes)   # [B,Q,C], [B,Q]
        in_warmup = self.epoch < self.warmup_epochs
        if pred_logits.requires_grad and (not in_warmup):
            target_detached = target.detach()
            valid_mask_detached = valid_mask.detach()
            pred_logits.register_hook(
                self._build_collect_grad_hook(target_detached, valid_mask_detached)
            )

        prob = torch.sigmoid(pred_logits)
        pred_t = prob * target + (1.0 - prob) * (1.0 - target)    # [B,Q,C]

        # focal-style CE
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        if in_warmup:
            # warmup 阶段只使用普通 sigmoid focal loss
            cls_loss = ce_loss * torch.pow((1.0 - pred_t), self.focal_gamma)
        else:
            # 类别相关 gamma
            map_val = 1.0 - self.pos_neg.detach()                      # [C]
            dy_gamma = self.focal_gamma + self.scale_factor * map_val  # [C]

            ff = dy_gamma.view(1, 1, C)                                # focusing factor
            wf = (dy_gamma / self.focal_gamma).view(1, 1, C)           # weighting factor
            cls_loss = ce_loss * torch.pow((1.0 - pred_t), ff.detach()) * wf.detach()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
            cls_loss = alpha_t * cls_loss

        # 只统计有效 query
        valid_mask_3d = valid_mask.unsqueeze(-1).float()
        cls_loss = cls_loss * valid_mask_3d

        if normalizer is None:
            # 你也可以改成正样本数，或匹配到的GT数
            normalizer = valid_mask.sum().clamp(min=1).float()

        loss = cls_loss.sum() / normalizer

        return loss * self.loss_weight


    def reset_collect_stats(self):
        self.tmp_pos_grad.zero_()
        self.tmp_neg_grad.zero_()
        self.cur_collect_idx = 0

    def _build_collect_grad_hook(self, target, valid_mask):
        def _hook(grad_in):
            self.collect_grad(grad_in, target, valid_mask)
            return grad_in

        return _hook

    def collect_grad(self, grad_in, target, valid_mask):
        assert grad_in.dim() == 3, f"expect [B, Q, C], but got {grad_in.shape}"
        assert grad_in.shape[-1] == self.num_classes, f"expect {self.num_classes} classes, but got {grad_in.shape[-1]}"
        assert target.shape == grad_in.shape, f"target shape mismatch: {target.shape} vs {grad_in.shape}"
        assert valid_mask.shape == grad_in.shape[:2], f"valid_mask shape mismatch: {valid_mask.shape} vs {grad_in.shape[:2]}"

        # grad: [B, Q, C]
        grad = grad_in.detach().abs()
        vm = valid_mask.unsqueeze(-1).float()

        pos_grad = (grad * target * vm).sum(dim=(0, 1))
        neg_grad = (grad * (1.0 - target) * vm).sum(dim=(0, 1))

        self.tmp_pos_grad += pos_grad
        self.tmp_neg_grad += neg_grad
        self.cur_collect_idx += 1

    def finalize_backward(self):
        """
        在一次 loss.backward() 结束后统一更新类别级统计。
        """
        if self.epoch < self.warmup_epochs:
            return None
        if self.cur_collect_idx == 0:
            return None

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.tmp_pos_grad)
            dist.all_reduce(self.tmp_neg_grad)

        momentum = 0.95

        # 第一次更新时避免全 0 历史导致统计过慢
        if (self.pos_grad.sum() + self.neg_grad.sum()) == 0:
            self.pos_grad.copy_(self.tmp_pos_grad)
            self.neg_grad.copy_(self.tmp_neg_grad)
        else:
            self.pos_grad.mul_(momentum).add_(self.tmp_pos_grad, alpha=1.0 - momentum)
            self.neg_grad.mul_(momentum).add_(self.tmp_neg_grad, alpha=1.0 - momentum)

        self.pos_neg = torch.clamp(
            self.pos_grad / (self.neg_grad + 1e-10),
            min=0.0,
            max=1.0
        )

        self.reset_collect_stats()

        return self.pos_neg.detach()