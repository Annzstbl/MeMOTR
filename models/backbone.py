# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
# @Description  : 用于 backbone 的设计和搭建
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from torchvision.models.efficientnet import Union
from utils.nested_tensor import NestedTensor

from .position_embedding import build as build_position_embedding
from hsmot.modules.conv import ConvMSI, ConvMSI_SE




class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Code reference: https://github.com/megvii-research/MOTR/blob/main/models/backbone.py
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """
    ResNet with frozen BatchNorm as backbone.
    """
    def __init__(self, backbone_name: str, train_backbone: bool, return_interm_layers: bool, input_channel: int, stem=None):
        """
        初始化一个 Backbone

        Args:
            backbone_name: backbone_name, only resnet50 is supported.
            train_backbone: whether finetune this backbone.
            return_interm_layers: whether return the intermediate layers' outputs.
        """
        super(Backbone, self).__init__()
        assert backbone_name == "resnet50", f"Backbone do not support '{backbone_name}'."
        backbone = resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or ("layer2" not in name and "layer3" not in name and "layer4" not in name): #如果训练backbone, 只训练layer2 layer3 layer4
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {
                "layer2": "0",
                "layer3": "1",
                "layer4": "2"
            }
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4", "0"}
            self.strides = [32]
            self.num_channels = [2048]

        def _build_stem(backbone):
            """
            Build the stem of the backbone.
            """
            if input_channel !=3:
                conv1_3ch = backbone.conv1
                if stem:
                    if stem == "conv3d":
                        # conv3d stem
                        return ConvMSI(
                            c1=1,
                            c2=conv1_3ch.out_channels,
                            c3=input_channel,
                            k=(3, *conv1_3ch.kernel_size),
                            s=(1, *conv1_3ch.stride),
                            p=(1, *conv1_3ch.padding),
                            groups=conv1_3ch.out_channels,
                            final_bn=False,
                            final_act=False,
                            use_bn_3d=False
                        )
                    elif stem == "conv3d_se" or stem == "conv3d_se_v2" or stem == "conv3d_se_v3":
                        return ConvMSI_SE(
                            c1=1,
                            c2=conv1_3ch.out_channels,
                            c3=input_channel,
                            k=(3, *conv1_3ch.kernel_size),
                            s=(1, *conv1_3ch.stride),
                            p=(1, *conv1_3ch.padding),
                            final_bn=False,
                            final_act=False,
                            use_bn_3d=False,
                            reduction=2
                        )

                else:
                    # conv2d stem
                    return nn.Conv2d(
                        in_channels=input_channel,
                        out_channels=conv1_3ch.out_channels,
                        kernel_size=conv1_3ch.kernel_size,
                        stride=conv1_3ch.stride,
                        padding=conv1_3ch.padding,
                        dilation=conv1_3ch.dilation,
                        groups=conv1_3ch.groups,
                        bias=(conv1_3ch.bias is not None)
                    )
            else:
                return backbone.conv1

        self.stem_conv = _build_stem(backbone)
        del backbone.conv1

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, ntensor: NestedTensor):
        x, se_weights = self.stem_conv(ntensor.tensors)
        outputs = self.backbone(x)

        res: Dict[str, NestedTensor] = dict()
        for name, output in outputs.items():
            masks = ntensor.masks
            assert masks is not None, "Masks should be NOT NONE."
            masks = F.interpolate(masks[None].float(), mode="nearest", size=output.shape[-2:]).to(masks.dtype)[0]
            res[name] = NestedTensor(output, masks)
        return res, se_weights


class BackboneWithPE(nn.Module):
    """
    Backbone with Position Embedding.
    Input: NestedTensor in (B, C, H, W)
    Output: Multi layer (B, C, H, W) as Image Features, multi layer (B, 2*num_pos_feats, H, W) as Position Embedding.
    """
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super(BackboneWithPE, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, ntensor: NestedTensor) -> (List[NestedTensor], List[torch.Tensor]):
        backbone_outputs = self.backbone(ntensor)
        features: List[NestedTensor] = list()
        pos_embeds: List[torch.Tensor] = list()
        # Image Features
        for _, output in sorted(backbone_outputs.items()):
            features.append(output)
        # Position Embedding
        for feature in features:
            pos_embeds.append(self.position_embedding(feature))

        return features, pos_embeds     # (B, C, H, W), (B, 2*num_pos_feats, H, W)，C is different in different layers.

    def n_inter_layers(self):
        return len(self.strides)

    def n_inter_channels(self):
        return self.num_channels


class Backbone_PE_SpectralWeights(nn.Module):
    """
    Backbone with Position Embedding and Spectral Weights.
    输出: 多尺度特征、位置编码、每个尺度的spectral_weights（如SE权重）。
    """
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module, spectral_embedding: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.spectral_embedding = spectral_embedding

    def n_inter_layers(self):
        return len(self.strides)

    def n_inter_channels(self):
        return self.num_channels


    def forward(self, ntensor: NestedTensor):
        backbone_outputs, se_weights = self.backbone(ntensor)
        features: List[NestedTensor] = []
        pos_embeds: List[torch.Tensor] = []
        spectral_embeds: List[torch.Tensor] = []
        # 取特征
        for _, output in sorted(backbone_outputs.items()):
            features.append(output)
        # 位置编码
        for feature in features:
            pos_embeds.append(self.position_embedding(feature))
        # 构建spectral_weights_list
        for feature in features:
            spectral_embeds.append(self.spectral_embedding(se_weights, feature))

        return features, pos_embeds, spectral_embeds

class Backbone_PE_SpectralWeights(nn.Module):
    """
    Backbone with Position Embedding and Spectral Weights.
    输出: 多尺度特征、位置编码、每个尺度的spectral_weights（如SE权重）。
    """
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module, spectral_embedding: nn.Module, weights_version="v1"):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.spectral_embedding = spectral_embedding
        self.weights_version = weights_version

    def n_inter_layers(self):
        return len(self.strides)

    def n_inter_channels(self):
        return self.num_channels

    def forward_v1(self, ntensor: NestedTensor):
        backbone_outputs, se_weights = self.backbone(ntensor)
        features: List[NestedTensor] = []
        pos_embeds: List[torch.Tensor] = []
        spectral_embeds: List[torch.Tensor] = []
        # 取特征
        for _, output in sorted(backbone_outputs.items()):
            features.append(output)
        # 位置编码
        for feature in features:
            pos_embeds.append(self.position_embedding(feature))
        # 构建spectral_weights_list
        for feature in features:
            spectral_embeds.append(self.spectral_embedding(se_weights, feature))

        return features, pos_embeds, spectral_embeds

    def forward_v2(self, ntensor: NestedTensor):
        layers=["layer2", "layer3", "layer4"]
        backbone_outputs, se_weights = self.backbone(ntensor)
        features: List[NestedTensor] = []
        pos_embeds: List[torch.Tensor] = []
        spectral_embeds: List[torch.Tensor] = []
        # 取特征
        for _, output in sorted(backbone_outputs.items()):
            features.append(output)
        # 位置编码
        for feature in features:
            pos_embeds.append(self.position_embedding(feature))
        # 构建spectral_weights_list
        for feature, layer in zip(features, layers):
            spectral_embeds.append(self.spectral_embedding(se_weights, feature, layer=layer))

        return features, pos_embeds, spectral_embeds
    
    def forward(self, ntensor: NestedTensor):
        if self.weights_version == "v1":
            return self.forward_v1(ntensor)
        elif self.weights_version == "v2":
            return self.forward_v2(ntensor)
        elif self.weights_version == "v3":
            return self.forward_v2(ntensor)#复用
        else:
            raise ValueError(f"Unsupported weights_version: {self.weights_version}")


class SpectralEmbedding(nn.Module):
    def __init__(self):
        super(SpectralEmbedding, self).__init__()

    def forward(self, spectral_weights: torch.Tensor, ntensor: NestedTensor, mode: str = "avg") -> torch.Tensor:
        return self.spectral_embedding(spectral_weights, ntensor, mode)

    def spectral_embedding(self, spectral_weights: torch.Tensor, ntensor:NestedTensor, mode: str = "avg") -> torch.Tensor:
        """
        将stride=2的权重降采样到特征图分辨率，输出与特征图空间一致的权重。
        Args:
            spectral_weights: [B, 8, H/2, W/2]
            feature: [B, Feature_C, H/stride, W/stride]
            mask: [B, H, W] 或 [B, H/stride, W/stride]
            mode: 'avg' 或 'max'，降采样方式
        Returns:
            out_weights: [B, 8, H/stride, W/stride]
        """

        mask = ntensor.masks
        feature = ntensor.tensors

        B, C, H_feat, W_feat = feature.shape[0], spectral_weights.shape[1], feature.shape[2], feature.shape[3]
        # 1. 降采样权重
        if mode == "avg":
            out_weights = F.adaptive_avg_pool2d(spectral_weights, (H_feat, W_feat))
        elif mode == "max":
            out_weights = F.adaptive_max_pool2d(spectral_weights, (H_feat, W_feat))
        else:
            raise ValueError("mode should be 'avg' or 'max'")


        # 将padding区域权重置为0
        out_weights = out_weights.masked_fill(mask.unsqueeze(1), 0)
        return out_weights

class SpectralEmbeddingConv(nn.Module):
    def __init__(self, resnet_output_layer: list[str]):
        super(SpectralEmbeddingConv, self).__init__()

        self.resnet_output_layer = resnet_output_layer

        def _build_spectral_weights_module(layer: str):
            if layer == "layer2":
                # 由512+8->64->8
                return nn.Sequential(
                    nn.Conv2d(in_channels=512+8, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )
            elif layer == "layer3":
                return nn.Sequential(
                    nn.Conv2d(in_channels=1024+8, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )
            elif layer == "layer4":
                return nn.Sequential(
                    nn.Conv2d(in_channels=2048+8, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )
            elif layer == "layer_extra":
                return nn.Sequential(
                    nn.Conv2d(in_channels=256+8, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unsupported layer: {layer}")

        self.conv_list = nn.ModuleList([_build_spectral_weights_module(layer) for layer in self.resnet_output_layer])


    def forward(self, spectral_weights: torch.Tensor, ntensor: NestedTensor, mode: str = "avg", layer='layer_extra') -> torch.Tensor:
        return self.spectral_embedding(spectral_weights, ntensor, mode, layer)

    def spectral_embedding(self, spectral_weights: torch.Tensor, ntensor: NestedTensor, mode: str = "avg", layer='layer_extra') -> torch.Tensor:
        '''
            extra_layer: 用于处理最后一层特征
        '''
        mask = ntensor.masks
        feature = ntensor.tensors
        B, C, H_feat, W_feat = feature.shape[0], spectral_weights.shape[1], feature.shape[2], feature.shape[3]
        out_weights = F.adaptive_avg_pool2d(spectral_weights, (H_feat, W_feat))
        out_weights = out_weights.masked_fill(mask.unsqueeze(1), 0)
        out_weights = self.conv_list[self.resnet_output_layer.index(layer)](torch.cat([feature, out_weights], dim=1))
        return out_weights


class SpectralEmbeddingV3(nn.Module):
    def __init__(self, resnet_output_layer: list[str]):
        super(SpectralEmbeddingV3, self).__init__()

        self.resnet_output_layer = resnet_output_layer
        self.layer2channel = {"layer2": 512, "layer3": 1024, "layer4": 2048, "layer_extra": 256}

        def _build_spectral_weights_module(layer: str):
            return nn.Sequential(
                nn.Conv2d(in_channels=self.layer2channel[layer], out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )       

        def _build_film_module(layer: str):
            return nn.Conv2d(in_channels=8, out_channels=self.layer2channel[layer]*2, kernel_size=3, stride=1, padding=1)

        self.conv_list = nn.ModuleList([_build_spectral_weights_module(layer) for layer in self.resnet_output_layer])
        self.film_list = nn.ModuleList([_build_film_module(layer) for layer in self.resnet_output_layer])


    def forward(self, spectral_weights: torch.Tensor, ntensor: NestedTensor, mode: str = "avg", layer='layer_extra') -> torch.Tensor:
        return self.spectral_embedding(spectral_weights, ntensor, mode, layer)

    def spectral_embedding(self, spectral_weights: torch.Tensor, ntensor: NestedTensor, mode: str = "avg", layer='layer_extra') -> torch.Tensor:
        '''
            extra_layer: 用于处理最后一层特征
        '''
        mask = ntensor.masks
        feature = ntensor.tensors
        B, C, H_feat, W_feat = feature.shape[0], spectral_weights.shape[1], feature.shape[2], feature.shape[3]
        out_weights = F.adaptive_avg_pool2d(spectral_weights, (H_feat, W_feat))
        out_weights = out_weights.masked_fill(mask.unsqueeze(1), 0)

        film_weights = self.film_list[self.resnet_output_layer.index(layer)](out_weights)# [B, C*2, H, W]
        gamma, beta = torch.chunk(film_weights, 2, dim=1)
        stage_feature = (1+gamma) * feature + beta
        out_weights = self.conv_list[self.resnet_output_layer.index(layer)](stage_feature)
        return out_weights





def build(config: dict) -> Union[BackboneWithPE, Backbone_PE_SpectralWeights]:
    CONFIG_STEM=config["STEM"]
    position_embedding = build_position_embedding(config=config)
    backbone = Backbone(backbone_name=config["BACKBONE"], train_backbone=True, return_interm_layers=True, input_channel=config["INPUT_CHANNELS"], stem=CONFIG_STEM)
    if CONFIG_STEM=="conv3d_se":
        spectral_embedding = SpectralEmbedding()
        return Backbone_PE_SpectralWeights(backbone=backbone, position_embedding=position_embedding, spectral_embedding=spectral_embedding, weights_version="v1")
    elif CONFIG_STEM=="conv3d_se_v2":
        spectral_embedding = SpectralEmbeddingConv(resnet_output_layer=["layer2", "layer3", "layer4", "layer_extra"])
        return Backbone_PE_SpectralWeights(backbone=backbone, position_embedding=position_embedding, spectral_embedding=spectral_embedding, weights_version="v2")
    elif CONFIG_STEM=="conv3d_se_v3":
        spectral_embedding = SpectralEmbeddingV3(resnet_output_layer=["layer2", "layer3", "layer4", "layer_extra"])
        return Backbone_PE_SpectralWeights(backbone=backbone, position_embedding=position_embedding, spectral_embedding=spectral_embedding, weights_version="v3")
    else:
        return BackboneWithPE(backbone=backbone, position_embedding=position_embedding)



