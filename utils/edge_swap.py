import torch
import math
import torch.nn as nn
from hsmot.datasets.pipelines.channel import version_index_to_str


class EdgeSwap(nn.Module):

    @staticmethod
    def edge_swap(bboxes: torch.Tensor, version: str|torch.Tensor, img_shape: tuple) -> torch.Tensor:
        if type(version) != str:
            version = version_index_to_str(version)

        if version == 'le135':#硬编码
            return EdgeSwap._edge_swap_le135(bboxes, img_shape)
        else:
            raise NotImplementedError(f"Unsupported version: {version}")

    @staticmethod
    def _edge_swap_le135(bboxes: torch.Tensor, img_shape: tuple) -> torch.Tensor:
        """
        归一化旋转框到 le135 格式。
        
        Args:
            bboxes: Tensor, 形状为 [bs, n, 5] 或 [n, 5]，5个值分别是 x, y, w, h, theta
                    theta 当前是 [0, 1] 编码
        
        Returns:
            Tensor: 归一化后的 bboxes，形状与输入相同
        """
        # 保存原始形状
        origin_shape = bboxes.shape
        # 统一处理为 [N, 5]
        bboxes = bboxes.view(-1, 5)

        h_img, w_img = img_shape
        
        # 分离各个分量
        x = bboxes[:, 0:1]  # [N, 1]
        y = bboxes[:, 1:2]  # [N, 1]
        w = bboxes[:, 2:3]  # [N, 1]
        h = bboxes[:, 3:4]  # [N, 1]
        theta_encoded = bboxes[:, 4:5]  # [N, 1], [0, 1] 编码

        # 真实分量
        w_real = w * w_img
        h_real = h * h_img
        
        # 解码 theta: 从 [0, 1] 解码到弧度 [-pi/4, 3*pi/4]
        theta_rad = theta_encoded * math.pi - math.pi / 4
        
        # 找到 w < h 的索引
        swap_mask = (w_real < h_real).squeeze(-1)  # [N]
        
        # 对于 w < h 的 box，互换 w 和 h
        w_new = torch.where(swap_mask.unsqueeze(-1), h, w)
        h_new = torch.where(swap_mask.unsqueeze(-1), w, h)
        
        # 对于 w < h 的 box，theta + 90度 (pi/2)
        theta_rad = torch.where(swap_mask.unsqueeze(-1), theta_rad + math.pi / 2, theta_rad)

        theta_rad = (theta_rad + math.pi/4) % math.pi - math.pi/4

        # 归一化
        theta_rad = (theta_rad+math.pi/4) / math.pi

        
        # 拼接结果
        bboxes_new = torch.cat([x, y, w_new, h_new, theta_rad], dim=-1)
        
        # 恢复原始形状
        bboxes_new = bboxes_new.view(origin_shape)
        
        return bboxes_new
