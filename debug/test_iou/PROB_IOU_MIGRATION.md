# 将训练 IoU 从当前实现改为 Prob IoU 需修改的文件

## 结论概览

- **需要改动的只有 hsmot 下的 2 个文件**，MeMOTR 侧无需改任何代码。
- 当前训练用的 IoU 来自：
  1. **损失**：`hsmot.loss.loss.loss_rotated_iou_norm_bboxes1`（内部用 `diff_iou_rotated_2d`）
  2. **匹配 cost 与 track.iou**：`hsmot.util.dist.box_iou_rotated_norm_bboxes1`（内部用 mmcv `box_iou_rotated`）

将这两处改为使用 `hsmot.loss.prob_iou_loss` 中的 `probiou` / `batch_probiou` 即可。

---

## 1. `hsmot/hsmot/loss/loss.py`

**作用**：训练时 box 的 IoU 损失（`get_loss_box` 里用的就是它）。

**当前逻辑**：
- `loss_rotated_iou_norm_bboxes1(bboxes1, bboxes2, img_shape, version)`
- `bboxes1`：归一化预测框 (N, 5)，`bboxes2`：绝对坐标 GT (N, 5)
- 按 `version`/`img_shape` 把 `bboxes1` 反归一化到像素 + 弧度角
- 调用 `diff_iou_rotated_2d(bboxes1, bboxes2)` 得到 ious，返回 (N,)

**修改**：
- 在文件顶部增加：`from hsmot.loss.prob_iou_loss import probiou`
- 把「反归一化后」的 `diff_iou_rotated_2d(...)` 换成：
  - `ious = probiou(bboxes1, bboxes2, CIoU=False, eps=1e-7)`
- 接口保持不变：仍接收 (norm_boxes, abs_boxes, img_shape, version)，返回 (N,) ious。

---

## 2. `hsmot/hsmot/util/dist.py`

**作用**：
- 匹配时：`matcher.py` 里 `cost_giou = -box_iou_rotated_norm_bboxes1(out_bbox, tgt_bbox, ...)`，需要 (N, M) 的 IoU 矩阵
- 训练中给 track 赋值：`criterion.py` 里 `trackinstances.iou[...] = box_iou_rotated_norm_bboxes1(..., aligned=True)`，需要 (N,) 的逐对 IoU

**当前逻辑**：
- `box_iou_rotated_norm_bboxes1(bboxes1, bboxes2, img_shape, version, mode, aligned, clockwise)`
- `bboxes1` 归一化，`bboxes2` 绝对坐标；先对 `bboxes1` 反归一化，再调 mmcv `box_iou_rotated`。

**修改**：
- 在文件顶部增加：`from hsmot.loss.prob_iou_loss import probiou, batch_probiou`
- 反归一化逻辑不变（与 `loss.py` 中相同的 version/angle 处理）。
- 替换 IoU 计算：
  - **aligned=True**：`ious = probiou(bboxes1, bboxes2, CIoU=False, eps=1e-7)`，返回 (N,)
  - **aligned=False**：`ious = batch_probiou(bboxes1, bboxes2, eps=1e-7)`，返回 (N, M)
- 参数 `mode`、`clockwise` 在 prob_iou 中无对应，可保留接口但在实现里忽略，或文档注明“使用 prob_iou 时无效”。

---

## 3. MeMOTR 侧（无需改）

| 文件 | 使用方式 | 说明 |
|------|----------|------|
| `MeMOTR/models/criterion.py` | `loss_rotated_iou_norm_bboxes1`、`box_iou_rotated_norm_bboxes1` | 仅从 hsmot 导入并调用，接口不变则无需改 |
| `MeMOTR/models/matcher.py` | `box_iou_rotated_norm_bboxes1(out_bbox, tgt_bbox, ...)` | 同上 |
| `MeMOTR/log/logger.py`、`MeMOTR/utils/vis_train_loss.py` | 只记录/可视化 `box_giou_loss` 等 key | 与 IoU 实现无关，无需改 |

---

## 4. 格式与依赖说明

- **prob_iou 输入**：xywhr，像素坐标 + 弧度角，与当前「反归一化后」的 bbox 格式一致，无需再转换。
- **依赖**：`hsmot.loss.prob_iou_loss` 仅依赖 `torch`（及 `numpy` 用于 `batch_probiou` 的 numpy 输入），不依赖 mmcv 的 `diff_iou_rotated_2d` / `box_iou_rotated`。

---

## 5. 修改后建议自测

- 跑 1 个 step 或 1 个 epoch，确认 `box_giou_loss` 有正常数值且可反传。
- 若有单元测试（如 `test_prob_iou.py`），可顺带跑一遍确保 prob_iou 行为未变。
