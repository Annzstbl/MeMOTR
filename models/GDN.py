import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2**-18):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        pedestal = self.reparam_offset ** 2
        beta_bound = (self.beta_min + pedestal) ** 0.5
        gamma_bound = self.reparam_offset

        self.register_buffer("pedestal", torch.tensor(pedestal))
        self.register_buffer("beta_bound", torch.tensor(beta_bound))
        self.register_buffer("gamma_bound", torch.tensor(gamma_bound))

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs