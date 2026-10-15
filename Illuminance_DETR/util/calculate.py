import torch
import numpy as np

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=1)

def box_cxcywhd_to_xyzxyz(x):
    cx, cy, cz, w, h, d = x.unbind(-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cz - 0.5 * d,
         cx + 0.5 * w, cy + 0.5 * h, cz + 0.5 * d]
    return torch.stack(b, dim=-1)

def box_xyzxyz_to_cxcywhd(x):
    x1, y1, z1, x2, y2, z2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
         (x2 - x1), (y2 - y1), (z2 - z1)]
    return torch.stack(b, dim=-1)

def box_iou_3d(boxes1, boxes2):
    vol1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    inter_xyz1 = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    inter_xyz2 = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    inter_whd = (inter_xyz2 - inter_xyz1).clamp(min=0)
    
    intersection_volume = inter_whd[:, :, 0] * inter_whd[:, :, 1] * inter_whd[:, :, 2]

    union_volume = vol1[:, None] + vol2 - intersection_volume

    iou = intersection_volume / (union_volume + 1e-6)
    return iou, union_volume

def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def _xyzxyz_np(boxes_cxcywhd):
    """(N, 6) cxcywhd -> (N, 6) xyzxyz。numpy 版（box_cxcywhd_to_xyzxyz と同じ変換）。"""
    b = np.asarray(boxes_cxcywhd, dtype=np.float64).reshape(-1, 6)
    c, s = b[:, :3], b[:, 3:]
    return np.concatenate([c - 0.5 * s, c + 0.5 * s], axis=1)


def _iou_1_to_m(pred_xyzxyz, gt_xyzxyz):
    """1 つの予測箱と M 個の GT 箱の 3D IoU を返す (M,)。"""
    lo = np.maximum(pred_xyzxyz[:3], gt_xyzxyz[:, :3])
    hi = np.minimum(pred_xyzxyz[3:], gt_xyzxyz[:, 3:])
    whd = np.clip(hi - lo, 0.0, None)
    inter = whd[:, 0] * whd[:, 1] * whd[:, 2]
    vol_p = np.prod(pred_xyzxyz[3:] - pred_xyzxyz[:3])
    vol_g = np.prod(gt_xyzxyz[:, 3:] - gt_xyzxyz[:, :3], axis=1)
    return inter / (vol_p + vol_g - inter + 1e-6)


def generalized_box_iou_3d(boxes1, boxes2):
    """軸平行 3D ボックスの一般化 IoU (GIoU) を総当たりで返す。

    boxes1: (..., N, 6) xyzxyz,  boxes2: (..., M, 6) xyzxyz  ->  (..., N, M)
    先頭の次元はブロードキャストされるので、損失用の (N,6)x(M,6) にも
    マッチャ用の (B,nq,6)x(B,max_n,6) にもそのまま使える。

    GIoU = IoU - (C - U) / C。C は両者を含む最小の軸平行直方体の体積。
    重なりが無いと IoU の勾配は消えるが、GIoU は「どれだけ離れているか」を
    C 経由で見るので勾配が残る。DETR 原実装 (util/box_ops.generalized_box_iou)
    の 3D 版で、係数も原実装と同じ既定値 (--giou_loss_coef 2) を使う。
    """
    lo = torch.max(boxes1[..., :, None, :3], boxes2[..., None, :, :3])
    hi = torch.min(boxes1[..., :, None, 3:], boxes2[..., None, :, 3:])
    whd = (hi - lo).clamp(min=0)
    inter = whd[..., 0] * whd[..., 1] * whd[..., 2]

    vol1 = (boxes1[..., 3:] - boxes1[..., :3]).clamp(min=0).prod(-1)
    vol2 = (boxes2[..., 3:] - boxes2[..., :3]).clamp(min=0).prod(-1)
    union = vol1[..., :, None] + vol2[..., None, :] - inter
    iou = inter / (union + 1e-6)

    c_lo = torch.min(boxes1[..., :, None, :3], boxes2[..., None, :, :3])
    c_hi = torch.max(boxes1[..., :, None, 3:], boxes2[..., None, :, 3:])
    c_whd = (c_hi - c_lo).clamp(min=0)
    vol_c = c_whd[..., 0] * c_whd[..., 1] * c_whd[..., 2]

    return iou - (vol_c - union) / (vol_c + 1e-6)


def calculate_ap_3d(all_preds, all_gts, num_classes, iou_thresholds):
    """クラス別 AP（11 点補間・貪欲マッチ）と mAP を返す。

    IoU は「予測 1 件 × その画像の GT 全件」を閾値によらず 1 度だけ計算し、
    閾値ごとにはマッチ済みフラグを張り直すだけにしてある。旧実装（閾値×予測ごとに
    torch の小カーネルを呼ぶ Python ループ）と結果は完全に一致し、実測で約 54 倍速い。
    --ap_select で 1 フレームあたりの予測数が Q から Q*C に増えるため、この高速化が
    無いと評価が現実的な時間で終わらない。
    """
    # --- GT をクラス別・画像別にまとめる（xyzxyz に変換して持つ） ---
    gts_by_class = [dict() for _ in range(num_classes)]
    for gt in all_gts:
        image_id = gt['image_id']
        labels = _to_numpy(gt['labels']).reshape(-1)
        if labels.size == 0:
            continue
        boxes = _xyzxyz_np(_to_numpy(gt['boxes']))
        for c in np.unique(labels):
            c = int(c)
            if c >= num_classes:
                continue
            gts_by_class[c].setdefault(image_id, []).append(boxes[labels == c])
    for c in range(num_classes):
        gts_by_class[c] = {k: np.concatenate(v, axis=0) for k, v in gts_by_class[c].items()}

    # --- 予測をクラス別にまとめる ---
    preds_by_class = [[] for _ in range(num_classes)]
    for pred, gt in zip(all_preds, all_gts):
        scores = _to_numpy(pred['scores']).reshape(-1)
        if scores.size == 0:
            continue
        image_id = gt['image_id']
        labels = _to_numpy(pred['labels']).reshape(-1)
        boxes = _xyzxyz_np(_to_numpy(pred['boxes']))
        for score, label, box in zip(scores, labels, boxes):
            c = int(label)
            if c < num_classes:
                preds_by_class[c].append((float(score), image_id, box))

    ap_matrix = np.zeros((num_classes, len(iou_thresholds)))
    class_exists = np.zeros(num_classes, dtype=bool)

    for c in range(num_classes):
        gts = gts_by_class[c]
        total_gts = sum(len(v) for v in gts.values())
        if total_gts == 0:
            continue
        class_exists[c] = True

        preds = preds_by_class[c]
        if not preds:
            continue

        preds.sort(key=lambda x: -x[0])

        # 閾値に依らない「最も重なる GT の index と IoU」を先に出しておく。
        best_idx = np.full(len(preds), -1, dtype=np.int64)
        best_iou = np.zeros(len(preds))
        for i, (_, image_id, box) in enumerate(preds):
            gt_boxes = gts.get(image_id)
            if gt_boxes is None or len(gt_boxes) == 0:
                continue
            ious = _iou_1_to_m(box, gt_boxes)
            j = int(ious.argmax())
            best_idx[i], best_iou[i] = j, float(ious[j])

        pred_image_ids = [p[1] for p in preds]

        for i_thresh, iou_thresh in enumerate(iou_thresholds):
            matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
            tps = np.zeros(len(preds))
            for i, image_id in enumerate(pred_image_ids):
                j = best_idx[i]
                if j >= 0 and best_iou[i] >= iou_thresh and not matched[image_id][j]:
                    matched[image_id][j] = True
                    tps[i] = 1.0

            tp_cumsum = np.cumsum(tps)
            fp_cumsum = np.cumsum(1.0 - tps)

            recalls = tp_cumsum / (total_gts + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            ap = 0.0
            for t in np.arange(0., 1.1, 0.1):
                mask = recalls >= t
                ap += (precisions[mask].max() if mask.any() else 0.0) / 11.0

            ap_matrix[c, i_thresh] = ap

    class_aps = {}
    for c in range(num_classes):
        class_aps[c] = float(np.mean(ap_matrix[c, :])) if class_exists[c] else 0.0

    mAP = float(np.mean(ap_matrix[class_exists, :])) if np.sum(class_exists) > 0 else 0.0

    return mAP, class_aps
