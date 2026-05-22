import os
import shutil
import time
import copy

import torch
from torch._C import NoneType
import torch.nn as nn
import torch.distributed

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from utils.nested_tensor import tensor_list_to_nested_tensor_already_padded
from models.memotr import MeMOTR
from structures.track_instances import TrackInstances
from models.criterion import build as build_criterion, ClipCriterion
from models.utils import get_model, save_checkpoint, load_checkpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from log.logger import Logger, ProgressLogger
from log.log import MetricLog
from models.utils import load_pretrained_model
from models.loss.efl_loss_help import SimpleGradientCollector
from utils.vis_val import visualize_validation_metrics
from utils.vis_train_loss import visualize_train_loss


def _sync_cuda_for_timing(device: torch.device, enabled: bool):
    if enabled and torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _time_after_cuda_sync(device: torch.device, enabled: bool) -> float:
    _sync_cuda_for_timing(device=device, enabled=enabled)
    return time.perf_counter()


def _collect_trainable_params_without_grad(model: nn.Module) -> list[str]:
    """Collect names of trainable parameters that have no gradient after backward."""
    model_core = get_model(model)
    return [
        name for name, param in model_core.named_parameters()
        if param.requires_grad and param.grad is None
    ]


def _amp_grad_scaler(enabled: bool):
    """PyTorch 2.0+ 使用 torch.amp.GradScaler('cuda')；更早版本使用 torch.cuda.amp.GradScaler。"""
    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _to_plain(x):
    """递归把 TrackedConfig / dict / list / tuple 转为普通 Python 容器，便于深拷贝。"""
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_plain(v) for v in x)
    return x


def _apply_overrides(base, overrides: dict):
    """对 base 做深拷贝，再用 overrides 中的字段做浅覆盖；尽量保持 base 的类型（TrackedConfig 兼容）。

    覆盖规则：
    - 顶层 key 直接替换（包括嵌套 dict 整体替换）；这意味着如果 overrides 想改 ``SCEM.ENABLE``，
      需要写完整的 ``SCEM: {...}`` 块。这样设计与 yaml 顶层一致，便于理解。
    - ``ENABLE`` 控制字段不会被写入合并结果。
    """
    from utils.utils import TrackedConfig

    base_plain = _to_plain(base)
    overrides_plain = _to_plain(overrides) if overrides else {}
    overrides_plain.pop("ENABLE", None)
    merged = copy.deepcopy(base_plain)
    for k, v in overrides_plain.items():
        merged[k] = v
    if isinstance(base, TrackedConfig):
        return TrackedConfig(merged)
    return merged


def train(config: dict):
    """训练入口：检测 ``STAGE1`` 块，按需先跑阶段 1（DETR pretrain），再跑阶段 2（MOT finetune）。

    yaml 顶层字段被视为**阶段 2**配置（兼容旧 yaml；不设 STAGE1 时退化为单段训练）。
    ``STAGE1`` 嵌套块描述对阶段 1 的覆盖项，例如：

    .. code-block:: yaml

        STAGE1:
          ENABLE: True
          EPOCHS: 20
          ONLY_TRAIN_DETR: True
          BATCH_SIZE: 2
          SAMPLE_STEPS: []
          SAMPLE_LENGTHS: [1]
          SAMPLE_INTERVALS: [1]
          LR_DROP_MILESTONES: [12]

    两阶段衔接：
    - 阶段 1 输出到 ``OUTPUTS_DIR/stage1_detr``。
    - 阶段 2 自动 ``PRETRAINED_MODEL = stage1_detr/last.pth``、``RESUME = None``，
      输出到 ``OUTPUTS_DIR/stage2_mot``；optimizer/scheduler 完全重建。
    """
    stage1 = config.get("STAGE1", None)
    enable_two_stage = (
        isinstance(stage1, dict)
        and bool(stage1.get("ENABLE", False))
        and int(stage1.get("EPOCHS", 0)) > 0
    )

    if not enable_two_stage:
        _run_one_stage(config, stage_tag=None)
        return

    base_outputs_dir = config["OUTPUTS_DIR"]
    base_submit_dir = config.get("SUBMIT_DIR", base_outputs_dir)

    # 阶段 1：用 STAGE1 字段覆盖 yaml 顶层
    stage1_overrides = {k: v for k, v in dict(stage1).items() if k != "ENABLE"}
    stage1_config = _apply_overrides(config, stage1_overrides)
    stage1_config["OUTPUTS_DIR"] = os.path.join(base_outputs_dir, "stage1_detr")
    stage1_config["SUBMIT_DIR"] = os.path.join(base_submit_dir, "stage1_detr")
    # 防止 _run_one_stage 内意外地再次进入两段分支
    stage1_config["STAGE1"] = {"ENABLE": False}

    stage1_ckpt = _run_one_stage(stage1_config, stage_tag="stage1_detr")

    if is_distributed():
        torch.distributed.barrier()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if stage1_ckpt is None or not os.path.exists(stage1_ckpt):
        raise FileNotFoundError(
            f"Stage 1 last.pth not found at {stage1_ckpt}. Cannot proceed to stage 2."
        )

    # 阶段 2：保留 yaml 顶层字段，但用阶段 1 的权重作为 PRETRAINED_MODEL
    stage2_config = _apply_overrides(config, {})
    stage2_config["OUTPUTS_DIR"] = os.path.join(base_outputs_dir, "stage2_mot")
    stage2_config["SUBMIT_DIR"] = os.path.join(base_submit_dir, "stage2_mot")
    stage2_config["PRETRAINED_MODEL"] = stage1_ckpt
    stage2_config["RESUME"] = None
    stage2_config["STAGE1"] = {"ENABLE": False}

    _run_one_stage(stage2_config, stage_tag="stage2_mot")
    return


def _run_one_stage(config: dict, stage_tag: str | None = None) -> str | None:
    """跑完整的一段训练（原 ``train`` 函数主体）。

    返回该阶段产出的 ``last.pth`` 路径（如果存在），便于两段衔接。
    """
    train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True, use_buffered_write=True)
    if stage_tag is not None:
        banner = f"=========== Stage: {stage_tag} ==========="
        train_logger.show(head=banner)
        train_logger.write(head=banner, filename="log.txt", mode="a")
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    loss_label_type = str(config.get("LOSS_LABEL_TYPE", "sigmoid_focal_loss"))
    normalized_loss_label_type = loss_label_type.lower()
    valid_loss_label_types = {"sigmoid_focal_loss", "eql_lossv2_nobg", "efl_loss", "efl_loss_closure"}
    if normalized_loss_label_type not in valid_loss_label_types:
        raise ValueError(
            f"Unsupported LOSS_LABEL_TYPE '{loss_label_type}', only support "
            f"{sorted(valid_loss_label_types)}"
        )
    train_logger.show(head=f"LOSS_LABEL_TYPE={loss_label_type}")
    train_logger.write(head=f"LOSS_LABEL_TYPE={loss_label_type}", filename="log.txt", mode="a")
    if normalized_loss_label_type == "eql_lossv2_nobg":
        train_logger.show(head=f"LOSS_LABEL_EQLV2_NOBG={config.get('LOSS_LABEL_EQLV2_NOBG', {})}")
        train_logger.write(
            head=f"LOSS_LABEL_EQLV2_NOBG={config.get('LOSS_LABEL_EQLV2_NOBG', {})}",
            filename="log.txt",
            mode="a"
        )
    elif normalized_loss_label_type in {"efl_loss", "efl_loss_closure"}:
        train_logger.show(head=f"LOSS_LABEL_EFL={config.get('LOSS_LABEL_EFL', {})}")
        train_logger.write(
            head=f"LOSS_LABEL_EFL={config.get('LOSS_LABEL_EFL', {})}",
            filename="log.txt",
            mode="a"
        )

    set_seed(config["SEED"])

    model = build_model(config=config)

    # Load Pretrained Model
    if config["PRETRAINED_MODEL"] is not None:
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=True, logger=train_logger)

    # Data process
    dataset_train = build_dataset(config=config, split="train", logger=train_logger)
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                        batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

    # Criterion
    criterion = build_criterion(config=config)
    criterion.set_device(torch.device("cuda", distributed_rank()))

    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model, logger=train_logger)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # AMP: bf16 uses autocast only; fp16 uses autocast + GradScaler
    use_amp = bool(config.get("USE_AMP", False))
    amp_dtype_str = str(config.get("AMP_DTYPE", "bf16")).lower()
    if amp_dtype_str == "fp16":
        amp_dtype = torch.float16
    elif amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported AMP_DTYPE '{amp_dtype_str}', use 'bf16' or 'fp16'")
    scaler = _amp_grad_scaler(enabled=(use_amp and amp_dtype is torch.float16))
    train_logger.show(head=f"USE_AMP={use_amp}, AMP_DTYPE={amp_dtype_str}, GradScaler enabled={scaler.is_enabled()}")
    train_logger.write(
        head=f"USE_AMP={use_amp}, AMP_DTYPE={amp_dtype_str}, GradScaler enabled={scaler.is_enabled()}",
        filename="log.txt",
        mode="a",
    )

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME"] is not None:
        if config["RESUME_SCHEDULER"]:
            load_checkpoint(
                model=model,
                path=config["RESUME"],
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        else:
            load_checkpoint(
                model=model, path=config["RESUME"], states=train_states, scaler=scaler
            )
            for _ in range(train_states["start_epoch"]):
                scheduler.step()

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    if is_distributed():
        # bn3d转为sync
        # bn3d_in_conv1 = model.backbone.backbone.backbone.conv1.bn3d
        # sync_bn3d_in_conv1 = nn.SyncBatchNorm.convert_sync_batchnorm(bn3d_in_conv1)
        # model.backbone.backbone.backbone.conv1.bn3d = sync_bn3d_in_conv1

        # model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=True)
        model = DDP(module=model, device_ids=[distributed_rank()])

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]
    use_checkpoint = "USE_CHECKPOINT" in config and config["USE_CHECKPOINT"]
    save_checkpoint_enabled = "SAVE_CHECKPOINT" in config and config["SAVE_CHECKPOINT"]
    checkpoint_every_n_epochs = config["CHECKPOINT_EVERY_N_EPOCHS"] if "CHECKPOINT_EVERY_N_EPOCHS" in config else 1
    checkpoint_tail_epochs = config["CHECKPOINT_TAIL_EPOCHS"] if "CHECKPOINT_TAIL_EPOCHS" in config else 1
    # Backward-compatible fallback for old key.
    evaluate_every_n_epochs = config["EVALUATE_EVERY_N_EPOCHS"] if "EVALUATE_EVERY_N_EPOCHS" in config else (
        config["EVALUATE_PER_EPOCH"] if "EVALUATE_PER_EPOCH" in config else 0
    )
    evaluate_tail_epochs = config["EVALUATE_TAIL_EPOCHS"] if "EVALUATE_TAIL_EPOCHS" in config else 5
    evaluate_force_epochs = config["EVALUATE_FORCE_EPOCHS"] if "EVALUATE_FORCE_EPOCHS" in config else []
    evaluate_force_epochs = set(evaluate_force_epochs if evaluate_force_epochs is not None else [])

    # criterion hook收集梯度
    gradient_collector = None
    if normalized_loss_label_type == "efl_loss":
        if criterion.efl_loss is None:
            raise RuntimeError("LOSS_LABEL_TYPE='efl_loss' but criterion.efl_loss is not initialized.")
        class_embed_modules = get_model(model).class_embed
        if isinstance(class_embed_modules, nn.ModuleList):
            hook_modules = []
            seen_module_ids = set()
            for module in class_embed_modules:
                module_id = id(module)
                if module_id in seen_module_ids:
                    continue
                seen_module_ids.add(module_id)
                hook_modules.append(module)
        else:
            hook_modules = [class_embed_modules]

        # def debug_forward_hook(module, inp, out):
        #     print("[forward hook triggered]", module.__class__.__name__,
        #         "out shape:", out.shape if torch.is_tensor(out) else type(out))
        # debug_handles = []
        # for m in hook_modules:
        #     debug_handles.append(m.register_forward_hook(debug_forward_hook))

        gradient_collector = SimpleGradientCollector(
            target_modules=hook_modules,
            collect_func=criterion.efl_loss.collect_grad,
            grad_type='output'
        )
        train_logger.show(head=f"EFL gradient collector enabled, hooked {len(hook_modules)} class_embed layers.")
        train_logger.write(
            head=f"EFL gradient collector enabled, hooked {len(hook_modules)} class_embed layers.",
            filename="log.txt",
            mode="a"
        )
    elif normalized_loss_label_type == "efl_loss_closure":
        if criterion.efl_loss is None:
            raise RuntimeError("LOSS_LABEL_TYPE='efl_loss_closure' but criterion.efl_loss is not initialized.")
        train_logger.show(head="EFL closure mode enabled, using pred_logits.register_hook for grad collection.")
        train_logger.write(
            head="EFL closure mode enabled, using pred_logits.register_hook for grad collection.",
            filename="log.txt",
            mode="a"
        )


    # 打印config使用情况
    train_logger.show(head=f"config使用情况: {config.access_summary()}")
    train_logger.write(head=f"config使用情况: {config.access_summary()}", filename="log.txt", mode="a")
    train_logger.show(head=f"config未使用项: {config.unused_keys()}")
    train_logger.write(head=f"config未使用项: {config.unused_keys()}", filename="log.txt", mode="a")

    # log记录开始时间
    train_logger.show(head=f"训练开始 Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    train_logger.write(head=f"训练开始 Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", filename="log.txt", mode="a")

    # Training:
    for epoch in range(start_epoch, config["EPOCHS"]):
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                            batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

        if epoch >= config["ONLY_TRAIN_QUERY_UPDATER_AFTER"]:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[3]["lr"] = 0.0
        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")
        default_lr_idx = -1
        for _ in range(len(lr_names)):
            if lr_names[_] == "lr":
                default_lr_idx = _
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")

        # NO_GRAD_FRAMES 已被废弃：训练循环要求每一帧都计算梯度。
        # 若 config 中残留该字段且实际解析出 > 0 的值，会在 train_one_epoch 入口被 ValueError 拦截。
        no_grad_frames = None
        if "NO_GRAD_FRAMES" in config:
            no_grad_steps = config.get("NO_GRAD_STEPS", []) or []
            for i in range(len(no_grad_steps)):
                if epoch >= no_grad_steps[i]:
                    no_grad_frames = config["NO_GRAD_FRAMES"][i]
                    break
            if no_grad_frames is not None and int(no_grad_frames) > 0:
                raise ValueError(
                    f"NO_GRAD_FRAMES is no longer supported (got {no_grad_frames} at epoch {epoch}). "
                    "Please remove NO_GRAD_FRAMES / NO_GRAD_STEPS from your config."
                )

        sample_length = dataset_train.sample_length
        dynamic_use_checkpoint, dynamic_checkpoint_level = resolve_dynamic_checkpoint_policy(
            config=config,
            sample_length=sample_length,
            use_checkpoint=use_checkpoint
        )

        train_one_epoch(
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            # metric_log=train_metric_log,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            use_dab=config["USE_DAB"],
            multi_checkpoint=multi_checkpoint,
            no_grad_frames=no_grad_frames,
            dynamic_use_checkpoint=dynamic_use_checkpoint and use_checkpoint,
            dynamic_checkpoint_level=dynamic_checkpoint_level,
            only_train_detr=config["ONLY_TRAIN_DETR"],
            timing_sync_cuda=config.get("TIMING_SYNC_CUDA", True),
            timing_log_interval=config.get("TIMING_LOG_INTERVAL", 2),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        current_epoch = epoch + 1
        # 与下方 submit_during_train 触发条件保持一致：这些轮次会跑验证
        in_evaluate_tail = evaluate_tail_epochs > 0 and current_epoch > (config["EPOCHS"] - evaluate_tail_epochs)
        hit_evaluate_interval = evaluate_every_n_epochs > 0 and (current_epoch % evaluate_every_n_epochs == 0)
        hit_evaluate_force = current_epoch in evaluate_force_epochs
        will_evaluate = hit_evaluate_interval or in_evaluate_tail or hit_evaluate_force

        in_checkpoint_tail = checkpoint_tail_epochs > 0 and current_epoch > (config["EPOCHS"] - checkpoint_tail_epochs)
        hit_checkpoint_interval = checkpoint_every_n_epochs > 0 and (current_epoch % checkpoint_every_n_epochs == 0)
        # 在会执行验证的 epoch 一并保存 checkpoint，便于对齐评估权重（含 EVALUATE_FORCE_EPOCHS）
        should_save_epoch_checkpoint = hit_checkpoint_interval or in_checkpoint_tail or will_evaluate

        if multi_checkpoint is True:
            pass
        elif save_checkpoint_enabled:
            if should_save_epoch_checkpoint:
                checkpoint_path = os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth")
                if is_main_process():
                    save_checkpoint(
                        model=model,
                        path=checkpoint_path,
                        states=train_states,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    shutil.copy2(checkpoint_path, os.path.join(config["OUTPUTS_DIR"], "last.pth"))
        else:
            # 兼容原逻辑：关闭 SAVE_CHECKPOINT 时仍保留最后一个 epoch；另外在会验证的 epoch 也保存以对齐评测
            if epoch == config["EPOCHS"] - 1 or will_evaluate:
                checkpoint_path = os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth")
                if is_main_process():
                    save_checkpoint(
                        model=model,
                        path=checkpoint_path,
                        states=train_states,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    shutil.copy2(checkpoint_path, os.path.join(config["OUTPUTS_DIR"], "last.pth"))
            train_logger.show(head="SAVE_CHECKPOINT=False：不按周期保存；仍会在末 epoch 与执行验证的 epoch（含 EVALUATE_FORCE_EPOCHS）保存 checkpoint")
            train_logger.write(head="SAVE_CHECKPOINT=False：不按周期保存；仍会在末 epoch 与执行验证的 epoch（含 EVALUATE_FORCE_EPOCHS）保存 checkpoint", filename="log.txt", mode="a")

        # 评估触发规则：固定间隔 或 尾部全评估 或 强制指定轮次
        if hit_evaluate_interval or in_evaluate_tail or hit_evaluate_force:
            from submit_engine import submit_during_train
            submit_during_train(config=config, epoch=epoch, model=model, only_train_detr=config["ONLY_TRAIN_DETR"], train_logger=train_logger)

        train_logger.flush_buffers()

    if gradient_collector is not None:
        gradient_collector.remove()
        train_logger.show(head="EFL gradient collector removed.")
        train_logger.write(head="EFL gradient collector removed.", filename="log.txt", mode="a")

    # log记录结束时间
    train_logger.write(head=f"训练结束 End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", filename="log.txt", mode="a")
    train_logger.show(head=f"训练结束 End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    # 训练结束后可视化训练损失和验证指标
    if is_main_process():
        train_log_file = os.path.join(config["OUTPUTS_DIR"], "train", "log.txt")
        train_loss_fig_dir = os.path.join(config["OUTPUTS_DIR"], "train", "fig_loss")

        try:
            train_logger.show(head="开始生成训练损失可视化...")
            visualize_train_loss(
                log_file=train_log_file,
                output_dir=train_loss_fig_dir
            )
            train_logger.show(head="训练损失可视化完成")
        except Exception as e:
            train_logger.show(head=f"训练损失可视化失败: {str(e)}")

        if not config["ONLY_TRAIN_DETR"]:
            try:
                train_logger.show(head="开始生成验证指标可视化...")
                best_result = visualize_validation_metrics(
                    val_root_path=config["OUTPUTS_DIR"],
                    fig_path=os.path.join(config["OUTPUTS_DIR"], 'fig'),
                    data_split=config["SUBMIT_DATA_SPLIT"],
                )
                if best_result:
                    train_logger.show(head=f"最佳组合分数: Epoch {best_result['epoch']}, "
                                         f"Combined Score: {best_result['combined_score']:.4f}")
                train_logger.show(head="验证指标可视化完成")
            except Exception as e:
                train_logger.show(head=f"验证指标可视化失败: {str(e)}")
        else:
            train_logger.show(head="ONLY_TRAIN_DETR=True，跳过 visualize_validation_metrics")

    # 返回该阶段产出的 last.pth 路径（若不存在则返回 None；上层据此判断阶段衔接是否可行）
    last_ckpt_path = os.path.join(config["OUTPUTS_DIR"], "last.pth")
    return last_ckpt_path if os.path.exists(last_ckpt_path) else None


def train_one_epoch(model: MeMOTR, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ClipCriterion, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, use_dab: bool = False,
                    multi_checkpoint: bool = False,
                    no_grad_frames: int | None = None,
                    dynamic_use_checkpoint: bool = False,
                    dynamic_checkpoint_level: int | None = None,
                    only_train_detr: bool = False,
                    timing_sync_cuda: bool = True,
                    timing_log_interval: int = 2,
                    use_amp: bool = False,
                    amp_dtype: torch.dtype = torch.bfloat16,
                    scaler=None):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        logger: unified logger.
        accumulation_steps:
        use_dab:
        multi_checkpoint:
        no_grad_frames: 已废弃，必须为 None 或 0；若传入 > 0 的值，会立即抛出 ValueError。
            （目的：保证训练循环内每一帧都计算梯度。）
        only_train_detr: 仅用于日志展示；不再用于训练循环分支。
            （模型构建处通过 ``build_query_updater`` 决定是否构造 QueryUpdater；
            提交/评测阶段在 submit_engine 中自行分支。）

    Returns:
        None
    """
    if no_grad_frames is not None and int(no_grad_frames) > 0:
        raise ValueError(
            f"`no_grad_frames` is no longer supported in train_one_epoch (got {no_grad_frames}). "
            "Please remove NO_GRAD_FRAMES / NO_GRAD_STEPS from your config; "
            "all frames must run with grad."
        )

    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device
    timing_log_interval = max(1, int(timing_log_interval))

    model_core = get_model(model)
    if dynamic_use_checkpoint:
        if dynamic_checkpoint_level is not None and hasattr(model_core, "set_checkpoint_level"):
            model_core.set_checkpoint_level(dynamic_checkpoint_level)
        model_core.enable_checkpoint(True)
        logger.write(
            head=f"--Epoch={epoch} Settings: Enable using checkpoint (level={dynamic_checkpoint_level})",
            filename="log.txt",
            mode="a",
        )
        logger.show(head=f"--Epoch={epoch} Settings: Enable using checkpoint (level={dynamic_checkpoint_level})")
    else:
        model_core.enable_checkpoint(False)
        logger.write(head=f"--Epoch={epoch} Settings: Disable using checkpoint", filename="log.txt", mode="a")
        logger.show(head=f"--Epoch={epoch} Settings: Disable using checkpoint")

    info_only_train_detr = (
        f"--Epoch={epoch} Settings: only_train_detr={only_train_detr} "
        f"(unified training loop; no_grad branch disabled)"
    )
    logger.show(head=info_only_train_detr)
    logger.write(head=info_only_train_detr, filename="log.txt", mode="a")

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.perf_counter()

    data_start_timestamp = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)

    criterion.set_epoch(epoch)

    TrackInstances.set_static_properties(
        False,
        use_dab=use_dab,
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
    )

    for i, batch in enumerate(dataloader):
        # batch[keys][batches][sequentials]
        # img_metas 现在按帧从 batch 内取出 list[dict]（见下方 padding_img_metas），
        # 以便 criterion / matcher / postprocess 可以按 batch 内每个样本自己的 img_shape /
        # version 计算 loss / IoU。bs>1 + multi-scale 训练时同一 batch 内各样本尺寸可能不同。
        iter_start_timestamp = time.perf_counter()
        data_time = iter_start_timestamp - data_start_timestamp
        setup_start_timestamp = iter_start_timestamp
        tracks = TrackInstances.init_tracks(
            batch=batch,
            hidden_dim=get_model(model).hidden_dim,
            num_classes=get_model(model).num_classes,
            device=device,
        )
        criterion.init_a_clip(
            batch=batch,
            hidden_dim=get_model(model).hidden_dim,
            num_classes=get_model(model).num_classes,
            device=device,
        )
        setup_time = time.perf_counter() - setup_start_timestamp

        frame_prepare_time = 0.0
        model_forward_time = 0.0
        criterion_time = 0.0
        postprocess_time = 0.0

        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
        with amp_ctx:
            num_frames = len(batch["imgs"][0])
            for frame_idx in range(num_frames):
                stage_start = time.perf_counter()
                frame = [fs[frame_idx] for fs in batch["imgs"]]
                padding_img_metas = [fs[frame_idx] for fs in batch["img_metas"]]
                for f in frame:
                    f.requires_grad_(False)
                frame = tensor_list_to_nested_tensor_already_padded(
                    tensor_list=frame, frame_metas=padding_img_metas
                ).to(device)
                stage_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
                frame_prepare_time += stage_end - stage_start

                stage_start = stage_end
                res = model(frame=frame, tracks=tracks, heatmap=None)
                stage_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
                model_forward_time += stage_end - stage_start

                stage_start = stage_end
                previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                    model_outputs=res,
                    tracked_instances=tracks,
                    frame_idx=frame_idx,
                    img_metas=padding_img_metas,
                )
                stage_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
                criterion_time += stage_end - stage_start

                if frame_idx < num_frames - 1:
                    stage_start = stage_end
                    tracks = get_model(model).postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets, img_metas=padding_img_metas,
                    )
                    stage_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
                    postprocess_time += stage_end - stage_start

        stage_start = time.perf_counter()
        with amp_ctx:
            loss_dict, log_dict = criterion.get_mean_by_n_gts()
            loss, log_dict = criterion.get_sum_loss_dict(loss_dict=loss_dict, log_dict=log_dict)
        stage_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
        loss_reduce_time = stage_end - stage_start

        backward_start = stage_end
        loss = loss / accumulation_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if i == 0 and is_main_process():
            _sync_cuda_for_timing(device=device, enabled=timing_sync_cuda)
            params_without_grad = _collect_trainable_params_without_grad(model)
            if params_without_grad:
                grad_check_msg = (
                    f"--[Epoch={epoch}, Iter={i}] {len(params_without_grad)} trainable params "
                    f"without gradient after backward:\n  "
                    + "\n  ".join(params_without_grad)
                )
            else:
                grad_check_msg = (
                    f"--[Epoch={epoch}, Iter={i}] All trainable params received gradients after backward."
                )
            logger.show(head=grad_check_msg)
            logger.write(head=grad_check_msg, filename="log.txt", mode="a")

        efl_pos_neg = None
        if getattr(criterion, "label_loss_type", "") == "efl_loss_closure" and criterion.efl_loss is not None:
            efl_pos_neg = criterion.efl_loss.finalize_backward()
        backward_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
        backward_time = backward_end - backward_start

        optimizer_time = 0.0
        if (i + 1) % accumulation_steps == 0:
            optimizer_start = backward_end
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            optimizer_end = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)
            optimizer_time = optimizer_end - optimizer_start

        iter_end_timestamp = time.perf_counter()
        active_time = iter_end_timestamp - iter_start_timestamp
        iter_time = iter_end_timestamp - data_start_timestamp

        metric_log.update(name="total_loss", value=loss.item() * accumulation_steps)
        metric_log.update(name="time per iter", value=iter_time)
        metric_log.update(name="time per data", value=data_time)
        metric_log.update(name="time/setup", value=setup_time)
        metric_log.update(name="time/frame_prepare", value=frame_prepare_time)
        metric_log.update(name="time/forward", value=model_forward_time)
        metric_log.update(name="time/criterion", value=criterion_time)
        metric_log.update(name="time/postprocess", value=postprocess_time)
        metric_log.update(name="time/loss_reduce", value=loss_reduce_time)
        metric_log.update(name="time/backward", value=backward_time)
        metric_log.update(name="time/optimizer", value=optimizer_time)
        metric_log.update(name="time/active", value=active_time)
        # 主损失：所有不带 "aux" 和 "class" 的损失；写入 metric_log
        for log_k, (val, _) in log_dict.items():
            if ("aux" not in log_k) and ("class" not in log_k):
                metric_log.update(name=log_k, value=val)

        # 周期性日志（降低同步频率以避免 NCCL 超时）
        if i % timing_log_interval == 0:
            metric_log.sync()
            max_memory = torch.cuda.max_memory_allocated() // (1024 ** 2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            second_per_data = metric_log.metrics["time per data"].avg
            second_per_forward = metric_log.metrics["time/forward"].avg
            second_per_backward = metric_log.metrics["time/backward"].avg
            header = (
                f"--[Epoch={epoch}, Iter={i}, "
                f"{second_per_iter:.2f}s/iter, "
                f"{second_per_data:.2f}s/data, "
                f"{second_per_forward:.2f}s/fwd, "
                f"{second_per_backward:.2f}s/bwd, "
                f"{i}/{dataloader_len} iters, "
                f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                f"Max Memory={max_memory}MB]"
            )
            logger.show(head=header, log=metric_log)
            logger.write(head=header, log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

            if is_main_process() and efl_pos_neg is not None:
                pos_neg_str = ", ".join([f"{x:.4f}" for x in efl_pos_neg.detach().cpu().tolist()])
                logger.show(head=f"efl_pos_neg: {pos_neg_str}")
                logger.write(head="efl_pos_neg", log=pos_neg_str, filename="log.txt", mode="a")

        if multi_checkpoint and is_main_process():
            checkpoint_path = os.path.join(logger.logdir[:-5], f"checkpoint_{int(i // 100)}.pth")
            save_checkpoint(model=model, path=checkpoint_path)
            shutil.copy2(checkpoint_path, os.path.join(logger.logdir[:-5], "last.pth"))

        train_states["global_iters"] += 1
        data_start_timestamp = _time_after_cuda_sync(device=device, enabled=timing_sync_cuda)

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.perf_counter()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"--[Epoch: {epoch}, Total Time: {epoch_minutes}min]", log=metric_log)
    logger.write(
        head=f"--[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
        log=metric_log,
        filename="log.txt",
        mode="a",
    )
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")

    return


def resolve_dynamic_checkpoint_policy(config: dict, sample_length: int, use_checkpoint: bool) -> tuple[bool, int | None]:
    """
    根据 sample_length 解析动态 checkpoint 策略。
    兼容两种配置形式：
    1) 标量:
       - DYNAMIC_USE_CHECKPOINT_THRESHOLD: int/float
       - CHECKPOINT_LEVEL: int
    2) 分阶段:
       - DYNAMIC_USE_CHECKPOINT_THRESHOLD: List[int/float]
       - CHECKPOINT_LEVEL: List[int]（与 threshold 一一对应）
    """
    if not use_checkpoint:
        return False, None

    thresholds_cfg = config.get("DYNAMIC_USE_CHECKPOINT_THRESHOLD", None)
    levels_cfg = config.get("CHECKPOINT_LEVEL", 1)

    if isinstance(thresholds_cfg, (list, tuple)):
        thresholds = [float(x) for x in thresholds_cfg]
        if len(thresholds) == 0:
            return False, None

        if isinstance(levels_cfg, (list, tuple)):
            levels = [int(x) for x in levels_cfg]
            if len(levels) != len(thresholds):
                raise ValueError(
                    "When DYNAMIC_USE_CHECKPOINT_THRESHOLD is a list, CHECKPOINT_LEVEL must be a list "
                    "with the same length."
                )
        else:
            levels = [int(levels_cfg)] * len(thresholds)

        stage_pairs = sorted(zip(thresholds, levels), key=lambda x: x[0])
        selected_level = None
        for threshold, level in stage_pairs:
            if sample_length >= threshold:
                selected_level = level
            else:
                break

        if selected_level is None:
            return False, None
        return True, max(1, min(3, int(selected_level)))

    threshold = float(thresholds_cfg)
    if sample_length >= threshold:
        if isinstance(levels_cfg, (list, tuple)):
            selected_level = int(levels_cfg[-1]) if len(levels_cfg) > 0 else 1
        else:
            selected_level = int(levels_cfg)
        return True, max(1, min(3, selected_level))
    return False, None


def get_param_groups(config: dict, model: nn.Module, logger: Logger = None) -> Tuple[List[Dict], List[str]]:
    """
    用于针对不同部分的参数使用不同的 lr 等设置
    Args:
        config: 实验的配置信息
        model: 需要训练的模型

    Returns:
        params_group: a list of params groups.
        lr_names: a list of params groups' lr name, like "lr_backbone".
    """
    def match_keywords(name: str, keywords: List[str]):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["backbone.backbone"]
    points_keywords = ["reference_points", "sampling_offsets"]  # 在 transformer 中用于选取参考点和采样点的网络参数关键字
    query_updater_keywords = ["query_updater"]
    dictionary_names = [] if "LR_DICTIONARY_NAMES" not in config else config["LR_DICTIONARY_NAMES"]
    _dictionary_scale = 1.0 if "LR_DICTIONARY_SCALE" not in config else config["LR_DICTIONARY_SCALE"]


    param_groups = [
        {   # backbone 学习率设置
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad and not match_keywords(n, dictionary_names)],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, points_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, dictionary_names) and p.requires_grad],
            "lr": config["LR"] * _dictionary_scale
        },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and not match_keywords(n, dictionary_names)
                       and p.requires_grad],
            "lr": config["LR"]
        }
    ]

    param_names = [
        {   # backbone 学习率设置
            "params": [n for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad and not match_keywords(n, dictionary_names)],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [n for n, p in model.named_parameters() if match_keywords(n, points_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [n for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [n for n, p in model.named_parameters() if match_keywords(n, dictionary_names) and p.requires_grad],
            "lr": config["LR"] * _dictionary_scale
        },
        {
            "params": [n for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and not match_keywords(n, dictionary_names)
                       and p.requires_grad],
            "lr": config["LR"]
        }
    ]
    
    #把所有参数的尺寸和requireds_grad,lr打印出来，提供debug
    for p in model.named_parameters():
        #通过param_names获得lr
        lr = "Not set"
        for param_group in param_names:
            if p[0] in param_group["params"]:
                lr = param_group.get("lr", "Not set")
                break
        logger.write(head=f'{p[0]}: {p[1].shape}, requires_grad={p[1].requires_grad}, lr={lr}', filename="log.txt", mode="a")
        logger.show(head=f'{p[0]}: {p[1].shape}, requires_grad={p[1].requires_grad}, lr={lr}')
    logger.flush_buffers()

    if logger is not None:
        logger.write(head=f"lr_dict param:, {param_names[3]['params']}", filename="log.txt", mode="a")

    # 打印 param_groups的所有lr和参数名
    for i, param_group in enumerate(param_names):
        lr = param_group.get("lr", "Not set")
        logger.write(head=f'=== Group {i+1} (lr={lr}) ===', filename="log.txt", mode="a")
        logger.write(head=f'Parameters ({len(param_group["params"])}):', filename="log.txt", mode="a")
        logger.show(head=f'=== Group {i+1} (lr={lr}) ===')
        logger.show(head=f'Parameters ({len(param_group["params"])}):')
        for j, param in enumerate(param_group["params"]):
            logger.write(head=f'  {j+1:3d}. {param}', filename="log.txt", mode="a")
            logger.show(head=f'  {j+1:3d}. {param}')

    return param_groups, ["lr_backbone", "lr_points", "lr_query_updater", "lr", "lr_dictionary"]
