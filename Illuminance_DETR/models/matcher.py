import torch
from scipy.optimize import linear_sum_assignment
from util.calculate import box_cxcywhd_to_xyzxyz, generalized_box_iou_3d
from torch import nn
from torch.nn.utils.rnn import pad_sequence

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_coord: float = 1, task: str = 'bbox',
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        # DETR 原実装の cost_giou 相当。損失に GIoU を入れるなら、割り当ても
        # 同じ尺度で行わないと「損失が下げたい組」と「マッチャが選ぶ組」がずれる。
        self.cost_giou = cost_giou if task == 'bbox' else 0.0
        self.task = task
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        # DETR 原実装はバッチ全予測×全正解の巨大コスト行列を作り対角ブロックだけ使うため、
        # batch_size に対して O(B^2) になる。ここでは各サンプル自身の (nq, n_b) ブロックのみを
        # padded テンソルで一括計算する（数値的には原実装の対角ブロックと完全に一致する）。
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].softmax(-1)              # (B, nq, C)

        if self.task == 'pose':
            out_coord = outputs["pred_keypoints"].flatten(2)      # (B, nq, 42)
            tgt_key = "keypoints"
        else:
            out_coord = outputs["pred_boxes"]                     # (B, nq, 6)
            tgt_key = "boxes"

        sizes = [len(v["labels"]) for v in targets]
        max_n = max(sizes) if sizes else 0

        if max_n == 0:
            empty = torch.as_tensor([], dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]

        # padding: 空サンプル (n_b=0) を含めて (B, max_n[, D]) を一括構築する
        tgt_labels_pad = pad_sequence([v["labels"] for v in targets],
                                      batch_first=True)                 # (B, max_n)
        tgt_coord_pad = pad_sequence([v[tgt_key].flatten(1) for v in targets],
                                     batch_first=True)                  # (B, max_n, D)

        # cost_class[b, q, j] = -out_prob[b, q, label[b, j]]
        idx = tgt_labels_pad.unsqueeze(1).expand(-1, num_queries, -1)   # (B, nq, max_n)
        cost_class = -torch.gather(out_prob, 2, idx)                    # (B, nq, max_n)
        cost_coord = torch.cdist(out_coord, tgt_coord_pad, p=1)         # (B, nq, max_n)

        C = self.cost_coord * cost_coord + self.cost_class * cost_class  # (B, nq, max_n)

        if self.cost_giou:
            # パディング分 (n_b < max_n) はここでは計算だけして、下の C[b, :, :n] で捨てる。
            giou = generalized_box_iou_3d(box_cxcywhd_to_xyzxyz(out_coord),
                                          box_cxcywhd_to_xyzxyz(tgt_coord_pad))
            C = C + self.cost_giou * (-giou)

        if torch.isnan(C).any() or torch.isinf(C).any():
            C = torch.nan_to_num(C, nan=1e5, posinf=1e5, neginf=-1e5)
        C = C.cpu()

        indices = []
        for b in range(bs):
            n = sizes[b]
            i, j = linear_sum_assignment(C[b, :, :n])
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))
        return indices

def build_matcher(args):
    task = args.task
    cost_class = args.set_cost_class
    
    if task == 'pose':
        # 既定 (None) では従来どおり損失係数を流用する。--set_cost_keypoint を
        # 指定すると、損失の重みを変えずにマッチングの幾何コストだけ動かせる。
        cost_coord = getattr(args, 'set_cost_keypoint', None)
        if cost_coord is None:
            cost_coord = args.keypoint_loss_coef
    else:
        cost_coord = args.set_cost_bbox

    cost_giou = getattr(args, 'set_cost_giou', 2.0) if getattr(args, 'giou', True) else 0.0

    return HungarianMatcher(cost_class=cost_class, cost_coord=cost_coord, task=task,
                            cost_giou=cost_giou)