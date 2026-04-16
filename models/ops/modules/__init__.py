# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


from .ms_deform_attn import MSDeformAttn
from .ms_deform_attn_rotate import MSDeformAttn_Rotate
from .ms_deform_attn_spectral import MSDeformAttnSpectral
from .ms_deform_attn_global import MSDeformAttnGlobal
from .ms_deform_attn_20260317 import MSDeformAttn20260317
from .ms_deform_attn_20260414 import MSDeformAttn20260414
from .ms_deform_attn_20260416 import MSDeformAttnAddTokenSharedLogits as MSDeformAttn20260416