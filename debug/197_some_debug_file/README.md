# MeMOTR 调试工具

这个调试脚本可以帮助你：
1. 加载训练好的模型权重
2. 查看模型参数和缓冲区
3. 运行一次前向传播
4. 捕获中间层的输出
5. 将tensor保存为CSV文件

## 基本用法

```bash
python decoder_spectral.py \
  --train-config /path/to/train/config.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --data-root /path/to/DATA_ROOT \
  --dataset-name hsmot_8ch \
  --split val \
  --seq 0001
```

## 主要功能

### 1. 查看模型参数

使用 `--show-params` 参数来查看匹配的模型参数：

```bash
# 查看包含 "transformer" 的参数
--show-params transformer

# 查看多个关键词匹配的参数
--show-params transformer,query_updater,backbone

# 打印参数值（前20个元素）
--print-values

# 保存参数到CSV文件
--save-csv
--csv-output-dir ./my_tensors
```

### 2. 捕获中间层输出

使用 `--hook-modules` 来注册forward hook，捕获指定模块的输出：

```bash
# 捕获transformer decoder的输出
--hook-modules transformer.decoder

# 捕获多个模块的输出
--hook-modules transformer.decoder,query_updater,backbone

# 同时保存hook输出到CSV
--save-csv
```

#### 查找可用模块

如果不确定模块名称，可以使用 `--list-modules` 来查看所有可用的模块：

```bash
python decoder_spectral.py \
  --train-config /path/to/config.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --list-modules
```

这会列出模型中所有可hook的模块，包括深层嵌套的模块。

### 3. CSV文件保存

当使用 `--save-csv` 时，脚本会：

- 为每个tensor创建CSV文件
- 文件名包含tensor名称和形状信息
- 自动处理不同维度的tensor：
  - 1D: 单列保存
  - 2D: 矩阵形式保存
  - 3D+: 展平保存，第一行包含原始形状信息

#### CSV文件命名规则

```
{参数名}_{形状1}_{形状2}_{形状3}.csv
```

例如：
- `transformer_decoder_layers_0_self_attn_in_proj_weight_256_768.csv`
- `hook_transformer_decoder_1_100_256.csv`

## 完整示例

### 1. 查看所有可用模块

```bash
python decoder_spectral.py \
  --train-config /data/users/litianhao/hsmot_code/MeMOTR/configs_hsmot_spectral_embed_197/08_train_hsmot8ch_spectralEmbV2_SpectralDecoderRefine2_Matcher_fconv10lr.yaml \
  --checkpoint /data4/litianhao/hsmot/memotr/spectralemb/07_v2_SpectralDecoderRefine_2gpu/checkpoint_17.pth \
  --list-modules
```

### 2. 调试特定模块

```bash
python decoder_spectral.py \
  --train-config /data/users/litianhao/hsmot_code/MeMOTR/configs_hsmot_spectral_embed_197/08_train_hsmot8ch_spectralEmbV2_SpectralDecoderRefine2_Matcher_fconv10lr.yaml \
  --checkpoint /data4/litianhao/hsmot/memotr/spectralemb/07_v2_SpectralDecoderRefine_2gpu/checkpoint_17.pth \
  --data-root /data/users/litianhao/hsmot_code/data \
  --dataset-name hsmot_8ch \
  --split train \
  --seq data37-9 \
  --show-params anchor,transformer \
  --print-values \
  --save-csv \
  --csv-output-dir ./debug_outputs \
  --hook-modules transformer.decoder,query_updater \
  --device cuda:0
```

## 输出说明

脚本会输出：

1. **参数信息**: 匹配参数的形状、类型、设备、元素数量
2. **CSV保存路径**: 如果启用了CSV保存，会显示每个tensor的保存路径
3. **模型输出**: 前向传播的最终输出（形状和类型）
4. **Hook输出**: 捕获的中间层输出（形状和类型）

## 注意事项

- 确保有足够的磁盘空间保存CSV文件
- 大型tensor的CSV文件可能很大
- 对于高维tensor，CSV文件会包含展平后的数据
- 使用 `--device cpu` 可以避免GPU内存不足的问题 