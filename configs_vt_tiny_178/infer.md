# VT-Tiny-MOT 推理与评测

## Submit 推理

```bash
cd /data1/users/litianhao01/hsmot/MeMOTR
python main.py --config-path configs_vt_tiny_99/20260511-1.yaml \
  --mode submit --submit-model checkpoint_17.pth \
  --submit-data-split test --submit-threads 4
```

输出：`{SUBMIT_DIR}/stage2_mot/test/tracker/{seq}.txt`

格式（正框 10 列）：`frame,id,x,y,w,h,score,cls,-1,-1`

## TrackEval 评测（独立运行）

```bash
cd /data1/users/litianhao01/hsmot/TrackEval
python scripts/run_vt_tiny_mot.py \
  --USE_PARALLEL False \
  --METRICS HOTA CLEAR Identity \
  --GT_COCO_ANN /data1/users/litianhao01/hsmot/data/VT-Tiny-MOT/annotations/instances_test2017.json \
  --IMG_FOLDER /data1/users/litianhao01/hsmot/data/VT-Tiny-MOT/test2017 \
  --TRACKERS_FOLDER /data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178/stage2_mot/test \
  --TRACKERS_TO_EVAL test \
  --TRACKER_SUB_FOLDER tracker \
  --IOU_THRESHOLD 0.2
```

- GT 来自 COCO JSON（`instances_{split}2017.json`），无需 mot txt
- 正框 IoU 匹配，与 HSMOT 旋转框评测（`hsmot_8ch.py`）完全分离
- 评测结果写到 `{TRACKERS_FOLDER}/test/eval/`

## 训练时 eval during train

训练触发 submit 后会自动调用 `run_vt_tiny_mot.py`；IoU 阈值默认取 yaml 中的 `TRACK_IOU_THRESH`（可用 `EVAL_IOU_THRESHOLD` 覆盖）。#TODO这是错的



# 可视化

python MeMOTR/utils/batch_vis_result.py \
    --txt_dir /data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178/stage2_mot/test/tracker \
    --img_root /data1/users/litianhao01/data/VT-Tiny-MOT/test2017 \
    --output_dir /data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178/stage2_mot/test/vis_results