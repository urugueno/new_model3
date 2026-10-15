import math
import os
import sys
from typing import Iterable

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

import util.misc as utils
import util.append
from util.calculate import box_cxcywhd_to_xyzxyz, box_iou_3d, calculate_ap_3d
from models.detr import select_detections

def _segment_mean(values, segment_id, num_segments):
    """サンプルごとに平均してから、サンプル間で平均する。

    旧実装（バッチ内を 1 サンプルずつ回して per-sample の mean を list に積み、
    最後に stack().mean() する）と同じ重み付けを保つためのもの。全ペアの
    単純平均とは異なり、マッチ数の多いサンプルを重く扱わない。
    """
    total = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    count = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    total.index_add_(0, segment_id, values)
    count.index_add_(0, segment_id, torch.ones_like(values))
    return (total / count).mean()


@torch.no_grad()
def compute_batch_metrics(outputs, targets, indices, task, args, device):
    """Hungarian マッチ済みペアから、ログ用メトリクスをバッチ一括で計算する。

    旧実装はバッチ内を 1 サンプルずつ Python で回しており、
      - bbox では 1 サンプルごとに box_iou_3d（小カーネル約 20 個）を呼ぶ
      - total_tp の .item() がサンプルごとに GPU→CPU 同期を強制する
      - pose では scale_fct をループ内で毎回 host→device 転送する
    ため、batch 2048 で実測 1.08 秒/iteration を消費していた（pose 側 0.21 秒）。
    ここは損失にも勾配にも一切関与しないログ専用の計算なので、全マッチペアを
    まとめて処理する。サンプル単位の平均の取り方（_segment_mean）と、
    ターゲット 0 件のサンプルを除外する挙動は旧実装と完全に同じ。
    """
    pred_logits = outputs['pred_logits']
    bs = pred_logits.shape[0]
    metrics = {}

    kept = [i for i, (_, tgt_idx) in enumerate(indices) if len(tgt_idx) > 0]

    total_tp = 0
    total_fn = 0

    if kept:
        # (サンプル内での平均) を取るためのセグメント ID。kept の並び順に 0..K-1。
        seg_id = torch.cat([torch.full((len(indices[i][1]),), k, dtype=torch.long, device=device)
                            for k, i in enumerate(kept)])
        b_idx = torch.cat([torch.full((len(indices[i][1]),), i, dtype=torch.long, device=device)
                           for i in kept])
        q_idx = torch.cat([indices[i][0].to(device) for i in kept])
        tgt_labels = torch.cat([targets[i]['labels'][indices[i][1]] for i in kept])
        num_seg = len(kept)

        pred_labels = pred_logits[b_idx, q_idx].argmax(-1)
        correct = (pred_labels == tgt_labels).to(pred_logits.dtype)

        metrics['accuracy'] = _segment_mean(correct, seg_id, num_seg)
        total_tp = correct.sum()          # .item() はループを抜けてから 1 回だけ

        if task == 'pose':
            tgt_kpts = torch.cat([targets[i]['keypoints'][indices[i][1]] for i in kept])
            pred_kpts = outputs['pred_keypoints'][b_idx, q_idx].reshape(tgt_kpts.shape)

            # 差分に対する変換なので origin（シフト）は相殺し、span だけ効く。
            # metric_scale は m -> ログ単位（既定 1000 で mm）。ループ外で 1 回だけ作る。
            scale_fct = (torch.as_tensor(args.space_size, device=device,
                                         dtype=pred_kpts.dtype) * args.metric_scale).view(1, 1, 3)

            diff = (pred_kpts - tgt_kpts) * scale_fct
            metrics['mpjpe'] = _segment_mean(diff.norm(dim=2).mean(1), seg_id, num_seg)
            diff_abs = diff.abs()
            for ax, name in enumerate(['mpjdle_x', 'mpjdle_y', 'mpjdle_z']):
                metrics[name] = _segment_mean(diff_abs[..., ax].mean(1), seg_id, num_seg)
        else:
            tgt_boxes = torch.cat([targets[i]['boxes'][indices[i][1]] for i in kept])
            pred_boxes = outputs['pred_boxes'][b_idx, q_idx]

            # 旧実装は (n, n) の IoU 行列を作って対角を取っていた。対応するペアの
            # IoU しか要らないので、ここでは要素ごとに直接計算する（値は同じ）。
            pb = box_cxcywhd_to_xyzxyz(pred_boxes)
            tb = box_cxcywhd_to_xyzxyz(tgt_boxes)
            whd = (torch.min(pb[:, 3:], tb[:, 3:]) - torch.max(pb[:, :3], tb[:, :3])).clamp(min=0)
            inter = whd[:, 0] * whd[:, 1] * whd[:, 2]
            vol_p = (pb[:, 3:] - pb[:, :3]).prod(-1)
            vol_t = (tb[:, 3:] - tb[:, :3]).prod(-1)
            metrics['iou'] = _segment_mean(inter / (vol_p + vol_t - inter + 1e-6), seg_id, num_seg)

        total_fn = sum(len(targets[i]['labels']) - len(indices[i][1]) for i in kept)

    background_class_idx = pred_logits.shape[-1] - 1
    num_predictions_as_object = (pred_logits.argmax(-1) != background_class_idx).sum()

    # ここまで total_tp / num_predictions_as_object はテンソルのまま持ち回り、
    # 同期は次の 1 行だけにする（旧実装はサンプルごとに .item() を呼んでいた）。
    total_tp = int(total_tp.item()) if torch.is_tensor(total_tp) else int(total_tp)
    total_fp = int(num_predictions_as_object.item()) - total_tp

    epsilon = 1e-6
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    if not np.isnan(f1_score):
        metrics['f1_score'] = f1_score

    return metrics


def uses_map_selection(args):
    """checkpoint 選択に val mAP を使うか。

    従来の (Acc+IoU+F1)/3 は mAP を一切見ておらず、実測では accuracy 項が
    スコアの変動にほぼ寄与していなかった（相関 -0.14〜+0.36）。さらに
    IoU-aware 分類 (--iou_aware) は「当たっていない予測の確信度を下げる」ので
    accuracy を意図的に下げる。従来指標のままだとその改善を悪化と読むため、
    bbox では val mAP で選ぶのを既定にする。pose は mAP を計算していないので従来どおり。
    """
    mode = getattr(args, 'select_metric', 'auto')
    if getattr(args, 'task', 'bbox') != 'bbox':
        return False
    return mode in ('auto', 'map')


def uses_mpjpe_selection(args):
    """checkpoint 選択とアーリーストッピングに val MPJPE だけを使うか（pose）。

    従来の -MPJPE/100 + Acc + F1 は、pose では acc / f1 が 0.99 以上で飽和して
    実質 MPJPE だけで決まっていた。条件によって F1 が動くと「F1 +0.01 = MPJPE 1 mm」
    の交換で MPJPE の悪いエポックを選びうるので、MPJPE に一本化するのを既定にする。
    """
    mode = getattr(args, 'select_metric', 'auto')
    if getattr(args, 'task', 'bbox') != 'pose':
        return False
    return mode in ('auto', 'mpjpe')


def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    # エポック冒頭の処理を持つ criterion（AWC の FIM 計算など）だけ呼ぶ
    begin_epoch = getattr(criterion, 'begin_epoch', None)
    if begin_epoch is not None:
        begin_epoch(model, data_loader, device, epoch)
    
    task = args.task
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    if task == 'pose':
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('mpjpe', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_x', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_y', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_z', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_keypoints', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))

    header = 'Epoch: [{}]'.format(epoch)

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        if isinstance(samples, dict):
            samples = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        else:
            samples = samples.to(device)
            
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if task == 'pose':
            metric_logger.update(loss=loss_value)
            if 'loss_keypoint' in loss_dict_scaled:
                metric_logger.update(loss_keypoints=loss_dict_scaled['loss_keypoint'])
        else:
            metric_logger.update(loss=loss_value)
            if 'loss_bbox' in loss_dict_scaled:
                metric_logger.update(loss_bbox=loss_dict_scaled['loss_bbox'])
        if 'loss' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss'])
        if 'loss_ce' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss_ce'])
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])

        with torch.no_grad():
            indices = criterion.matcher(outputs, targets)
            for name, value in compute_batch_metrics(
                    outputs, targets, indices, task, args, device).items():
                metric_logger.update(**{name: value})

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, 
             header_prefix: str = 'Test', prediction_filename: str = None, base_ds=None, args=None):
    model.eval()
    criterion.eval()

    task = args.task
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    if task == 'pose':
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('mpjpe', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_x', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_y', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('mpjdle_z', utils.SmoothedValue(window_size=1, fmt="{median:.3f} ({global_avg:.3f})"))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_keypoints', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('iou', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('f1_score', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1))
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1))

    header = f'{header_prefix}:'

    predictions_for_csv = []
    all_preds_for_ap = []
    all_gts_for_ap = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        if isinstance(samples, dict):
            samples = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        else:
            samples = samples.to(device)
            
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)

        selected_sensor_ids_batch = None
        selected_sensor_coords_batch = None
        
        if 'selected_sensor_indices' in outputs and outputs['selected_sensor_indices'] is not None:
            selected_sensor_ids_batch = outputs['selected_sensor_indices'].cpu().numpy()
            
            if 'coords' in samples:
                all_coords = samples['coords'].cpu().numpy()
                selected_sensor_coords_batch = []
                for i in range(len(selected_sensor_ids_batch)):
                    indices = selected_sensor_ids_batch[i]
                    coords = all_coords[i][indices]
                    selected_sensor_coords_batch.append(coords)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        metric_logger.update(loss=losses.item())

        if task == 'pose':
            if 'loss_keypoint' in loss_dict_scaled:
                metric_logger.update(loss_keypoints=loss_dict_scaled['loss_keypoint'])
        else:
            if 'loss_bbox' in loss_dict_scaled:
                metric_logger.update(loss_bbox=loss_dict_scaled['loss_bbox'])
        if 'loss' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss'])
        if 'loss_ce' in loss_dict_scaled:
            metric_logger.update(loss_ce=loss_dict_scaled['loss_ce'])
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])
        indices = criterion.matcher(outputs, targets)
        for name, value in compute_batch_metrics(
                outputs, targets, indices, task, args, device).items():
            metric_logger.update(**{name: value})

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_target_origins = torch.stack([t["orig_origin"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, orig_target_origins)

        if task == 'pose':
            for i, res in enumerate(results):
                sensor_coords_str = ""
                
                if selected_sensor_coords_batch is not None:
                    coords = selected_sensor_coords_batch[i]
                    coords_list = [f"({c[0]:.2f}, {c[1]:.2f})" for c in coords]
                    sensor_coords_str = str(coords_list).replace("'", "")

                util.append.append_pose_predictions(
                    predictions_for_csv, res, targets[i], sensor_coords_str
                )
        else:
            out_logits = outputs['pred_logits']
            out_bbox = outputs['pred_boxes']

            # mAP に載せる検出集合の作り方。既定 (flat_all) は Deformable DETR /
            # DINO と同じ「(クエリ, クラス) の全組を出す」後処理。--ap_select legacy で
            # 従来の「∅ argmax のクエリを捨てる」挙動に戻せる（過去結果の再現用）。
            # PostProcess 側は CSV 用に legacy のままなので、ここと役割が分かれている。
            scores, labels, query_idx, keep = select_detections(
                out_logits, getattr(args, 'ap_select', 'flat_all'), getattr(args, 'ap_topk', 20))

            for i in range(len(targets)):
                q = query_idx[i][keep[i]]
                all_preds_for_ap.append({
                    'scores': scores[i][keep[i]].cpu(),
                    'labels': labels[i][keep[i]].cpu(),
                    'boxes': out_bbox[i][q].cpu()
                })
                
                all_gts_for_ap.append({
                    'labels': targets[i]['labels'].cpu(),
                    'boxes': targets[i]['boxes'].cpu(),
                    'image_id': targets[i]['image_id'].item()
                })

            for i, res in enumerate(results):
                sensor_coords_str = ""
                
                if selected_sensor_ids_batch is not None:
                    if selected_sensor_coords_batch is not None:
                        coords = selected_sensor_coords_batch[i]
                        coords_list = [f"({c[0]:.2f}, {c[1]:.2f})" for c in coords]
                        sensor_coords_str = str(coords_list).replace("'", "")

                util.append.append_bbox_predictions(
                    predictions_for_csv, res, targets[i], device, sensor_coords_str
                )

    metric_logger.synchronize_between_processes()
    print("")

    if len(predictions_for_csv) > 0:
        if task == 'pose' and header_prefix == 'Test':
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"--- Final Evaluation Summary ---")
            print(f"\nOverall Pose Estimation Performance:")
            
            acc = stats.get('accuracy', float('nan'))
            f1 = stats.get('f1_score', float('nan'))
            mpjpe = stats.get('mpjpe', float('nan'))
            mpjdle_x = stats.get('mpjdle_x', float('nan'))
            mpjdle_y = stats.get('mpjdle_y', float('nan'))
            mpjdle_z = stats.get('mpjdle_z', float('nan'))

            print(f"MPJPE     : {mpjpe:.4f}")
            print(f"MPJDLE X  : {mpjdle_x:.4f}")
            print(f"MPJDLE Y  : {mpjdle_y:.4f}")
            print(f"MPJDLE Z  : {mpjdle_z:.4f}")
            print(f"Accuracy  : {acc:.4f}")
            print(f"F1 Score  : {f1:.4f}")
                
            print("-"*55 + "\n")

        elif task == 'bbox' and header_prefix != 'Test' and uses_map_selection(args):
            # checkpoint 選択用の val mAP。閾値 0.5:0.95 を 1 本だけ計算する
            # （calculate_ap_3d は IoU を閾値に依らず 1 度しか計算しないので、
            # 10 閾値でも 1 閾値とほぼ同じコスト。実測で 1 epoch あたり約 10 秒）。
            val_mAP, _ = calculate_ap_3d(
                all_preds_for_ap, all_gts_for_ap, criterion.num_classes,
                np.arange(0.5, 1.0, 0.05)
            )
            metric_logger.update(ap=val_mAP)

        elif task == 'bbox' and header_prefix == 'Test':
            mAP50, class_ap50 = calculate_ap_3d(
                all_preds_for_ap, all_gts_for_ap, criterion.num_classes, [0.50]
            )
            
            mAP75, class_ap75 = calculate_ap_3d(
                all_preds_for_ap, all_gts_for_ap, criterion.num_classes, [0.75]
            )

            ap_thresholds_avg = np.arange(0.5, 1.0, 0.05)
            mAP, class_ap = calculate_ap_3d(
                all_preds_for_ap, all_gts_for_ap, criterion.num_classes, ap_thresholds_avg
            )

            print(f"--- Final Evaluation Summary ---")
            print(f"\nOverall Object Detection Performance:  "
                  f"(ap_select={getattr(args, 'ap_select', 'flat_all')})")
            print(f"  mAP    (0.50:0.95): {mAP:.4f}")
            print(f"  mAP50  (0.50)     : {mAP50:.4f}")
            print(f"  mAP75  (0.75)     : {mAP75:.4f}")

            print(f"\nPer-Class Performance:")
            print(f"  {'Class ID':<10} | {'AP (0.5:0.95)':<15} | {'AP50':<10} | {'AP75':<10}")
            print("-" * 55)
            for c in range(criterion.num_classes):
                print(f"  {c:<10} | {class_ap.get(c, 0.0):<15.4f} | {class_ap50.get(c, 0.0):<10.4f} | {class_ap75.get(c, 0.0):<10.4f}")

            metric_logger.update(ap=mAP, ap50=mAP50, ap75=mAP75)
            for c in range(criterion.num_classes):
                metric_logger.update(**{
                    f'AP_class_{c}': class_ap.get(c, 0.0),
                    f'AP50_class_{c}': class_ap50.get(c, 0.0),
                    f'AP75_class_{c}': class_ap75.get(c, 0.0)
                })
            print("")

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if prediction_filename and output_dir:
            csv_path = os.path.join(output_dir, prediction_filename)
            pd.DataFrame(predictions_for_csv).to_csv(csv_path, index=False)

    return stats