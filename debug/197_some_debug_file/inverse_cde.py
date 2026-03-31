import torch
import scipy.stats as st

# 生成均匀分布样本
u = torch.rand(5)  # [0,1) 区间
print("Uniform:", u)

# 转换为高斯
g = torch.distributions.Normal(0,1).icdf(u)
print("Gaussian:", g)

# 或者用 scipy
import numpy as np
from scipy.stats import norm
u = np.random.rand(5)
g = norm.ppf(u)  # ppf = percent point function = inverse CDF
print("Gaussian:", g)
