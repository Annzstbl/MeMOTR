1. 确认track_instances结构中的个字段是如何流动的
2. 确认aux_loss是否不计算上一帧确定的目标
3. memotr模型初始化时候对spectral_refine的赋值



Q1 
    track_instances中的字段有
    ref_pts 
        其初始化和更新分为三种情况
        a. 之前跟踪到的目标: 在query_updater中满足socres>th iou>0.5的时候, 会由result中的boxes字段值更新。
        b. 这一帧匹配的目标: 由最后一层decoder layer的输入得到。同时如果在query_updater中满足socres>th iou>0.5的会由boxes更新。
        c. 未匹配到的目标，由init_ref_pts得到
    query_embed
        其更新分为两种情况
        a. 上一帧的目标由query_updater控制如何更新
        b. 这一帧匹配的目标和未匹配到的目标由["aux_outputs][-1]["queries"]得到, 其是最后一层decoder layer的输出.
    ids
    boxes
    labels
    logits
    matched_idx #只在criterion中使用
    output_embed #decoder的output_embed
    disappear_time
    scores
    area
    iou #匹配后计算
    last_output #新生的由output_embed赋值
    long_memory #新生的由query_embed赋值，之后不断更新
    last_appear_boxes #

    query_spectral_weights # 这一帧匹配的目标由["aux_outputs][-1]["queries"]
    pred_spectral_weights






函数说明
function update_tracked_instances() @criterion.py  
    更新上一帧的tracked_instnaces
    更新 boxes logits output_embed 
    重置 matched_idx labels
    query_embed和ref_pts不更新而是在query_updater中更新。



line 250
根据新匹配到的目标构建trackinstances
query_embed = model_outputs["aux_outputs"][-1]["queries"]
ref_pts = model_outputs["last_ref_pts"][b][output_idx]
output_embed = model_outputs["outputs"][b][output_idx]
boxes = model_outputs["pred_bboxes"][b][output_idx]
logits = model_outputs["pred_logits"][b][output_idx]
iou = torch.zeros((len(gt_idx),), dtype=torch.float)
spectral_weights = model_outputs["last_query_spectral_weights"][b][output_idx]



