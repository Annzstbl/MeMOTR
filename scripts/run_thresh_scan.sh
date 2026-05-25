#!/usr/bin/env bash
# =============================================================================
# MeMOTR 阈值参数分阶段扫描 — 便捷启动脚本
# 自动激活 conda hsmot 环境并调用 scan_threshold_params.py
#
# 完整说明:
#   bash scripts/run_thresh_scan.sh help
#   python scripts/scan_threshold_params.py --help
# =============================================================================
#
# 扫描流程（每阶段约 5 组，阶段间人工选最优）:
#   Stage 1: DET，RESULT = DET
#   Stage 2: TRACK (< DET)，RESULT = TRACK，需 --det
#   Stage 3: RESULT，需 --det --track
#   Stage 4: UPDATE_THRESH，需 --det --track --result
#
# 示例:
#   SCAN_ROOT=/data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178/stage2_mot/thresh_scan
#   COMMON="--config-path configs_vt_tiny_99/20260511-1.yaml \
#     --submit-dir /data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178 \
#     --submit-model checkpoint_17.pth \
#     --scan-root ${SCAN_ROOT} \
#     --available-gpus 0 --submit-threads 2 --iou-threshold 0.3"
#
#   bash scripts/run_thresh_scan.sh --stage 1 ${COMMON} --dry-run
#   bash scripts/run_thresh_scan.sh --stage 2 --det 0.5 ${COMMON}
#   bash scripts/run_thresh_scan.sh --stage 3 --det 0.5 --track 0.4 ${COMMON}
#   bash scripts/run_thresh_scan.sh --stage 4 --det 0.5 --track 0.4 --result 0.45 ${COMMON}
#
# 输出: {scan_root}/stage{N}/summary.md，每实验含 test/eval_00 与 test/eval_01
# =============================================================================

CONDA_ENV="${CONDA_ENV:-hsmot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMOTR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -euo pipefail

_conda_activate() {
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
}

if [[ "${1:-}" == "help" || "${1:-}" == "--usage" ]]; then
  _conda_activate
  cd "${MEMOTR_ROOT}"
  exec python scripts/scan_threshold_params.py --help
fi

_conda_activate
cd "${MEMOTR_ROOT}"
exec python scripts/scan_threshold_params.py "$@"
