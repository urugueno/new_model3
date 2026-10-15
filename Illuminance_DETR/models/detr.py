import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy)
from util.calculate import box_cxcywhd_to_xyzxyz, box_iou_3d, generalized_box_iou_3d

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

class EQCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        self.task = args.task
        self.pose_param = getattr(args, 'pose_param', 'abs') if self.task == 'pose' else None
        if self.pose_param == 'root_rel':
            # 先頭 3 次元がルート（腰中点）、残り 42 次元がルートからの相対オフセット
            self.coord_embed = MLP(hidden_dim, hidden_dim, 3 + 42, 3)
            offset_range = torch.as_tensor(args.space_size, dtype=torch.float32).reciprocal() \
                * args.pose_offset_range
            self.register_buffer('offset_range', offset_range, persistent=False)  # 正規化空間, (3,)
        elif self.task == 'pose':
            self.coord_embed = MLP(hidden_dim, hidden_dim, 42, 3)
        else:
            self.coord_embed = MLP(hidden_dim, hidden_dim, 6, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args

        self.env_query_cnn = None
        if self.args is not None:
            if self.args.environment == 'EQ':
                self.env_query_cnn = EQCNN(
                    input_dim=self.args.actual_num_sensors,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim
                )

    def forward(self, samples: NestedTensor):
        backbone_output = self.backbone(samples)
        selected_sensor_indices = None

        if len(backbone_output) == 4:
            features, pos, attn_mask, selected_sensor_indices = backbone_output
        elif len(backbone_output) == 3:
            features, pos, attn_mask = backbone_output
        else:
            features, pos = backbone_output
            attn_mask = None

        src, mask = features[-1].decompose()
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1], attn_mask=attn_mask)[0]

        if self.env_query_cnn is not None:
            env_vector = samples['env_vector']
            env_query_pe = self.env_query_cnn(env_vector)
            hs = hs + env_query_pe.unsqueeze(0).unsqueeze(2)

        outputs_class = self.class_embed(hs)
        coord_raw = self.coord_embed(hs)
        extra = {}
        if self.pose_param == 'root_rel':
            root = coord_raw[..., :3].sigmoid()                                   # (L, B, Q, 3)
            offset = coord_raw[..., 3:].tanh().unflatten(-1, (14, 3)) * self.offset_range
            outputs_coord = (root.unsqueeze(-2) + offset).flatten(-2)            # (L, B, Q, 42)
            extra = {'pred_root': root, 'pred_offset': offset.flatten(-2)}
        else:
            outputs_coord = coord_raw.sigmoid()

        if self.task == 'pose':
            outputs_class = torch.nan_to_num(outputs_class, nan=0.0, posinf=1e4, neginf=-1e4)
            outputs_coord = torch.nan_to_num(outputs_coord, nan=0.0, posinf=1.0, neginf=0.0)
            extra = {k: torch.nan_to_num(v, nan=0.0) for k, v in extra.items()}

        out_coord_key = 'pred_keypoints' if self.task == 'pose' else 'pred_boxes'
        layered = {'pred_logits': outputs_class, out_coord_key: outputs_coord, **extra}
        out = {k: v[-1] for k, v in layered.items()}

        if selected_sensor_indices is not None:
            out['selected_sensor_indices'] = selected_sensor_indices

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(layered)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, layered):
        num_layers = next(iter(layered.values())).shape[0]
        return [{k: v[i] for k, v in layered.items()} for i in range(num_layers - 1)]
    
@torch.no_grad()
def match_quality(outputs, targets, indices, idx, task, sigma=0.05):
    """マッチ済みペアごとの「当たり具合」 q ∈ [0, 1] を返す (n_matched,)。

    IoU-aware classification（Generalized Focal Loss / VarifocalNet）で、
    分類ターゲットの 1.0 を置き換えるために使う。q は必ず detach される
    （座標の勾配を分類経路に流すと学習が壊れるため）。

    bbox: 3D IoU そのもの。
    pose: COCO OKS と同じ形の exp(-d^2 / 2σ^2)。d は正規化空間での平均関節距離で、
          σ=0.05 は空間幅 3 m に対して 15 cm にあたる。
    """
    if task == 'pose':
        src = outputs['pred_keypoints'][idx]
        tgt = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src = src.reshape(src.shape[0], -1, 3)
        d = (src - tgt.reshape(src.shape)).norm(dim=-1).mean(-1)
        return torch.exp(-(d ** 2) / (2 * sigma ** 2)).clamp(0.0, 1.0)

    src = outputs['pred_boxes'][idx]
    tgt = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    iou, _ = box_iou_3d(box_cxcywhd_to_xyzxyz(src), box_cxcywhd_to_xyzxyz(tgt))
    return iou.diag().clamp(0.0, 1.0)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 iou_aware=True, quality_sigma=0.05, giou=True, root_loss_coef=14.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # 分類ターゲットを 1.0 ではなく「その予測の当たり具合 q」にする
        # （IoU-aware classification / Generalized Focal Loss・VarifocalNet と同じ発想）。
        # こうすると softmax の前景確率がそのまま位置精度の推定量になり、mAP の
        # score 順位づけが意味を持つ。q の作り方は _match_quality を参照。
        self.iou_aware = iou_aware
        self.quality_sigma = quality_sigma
        self.root_loss_coef = root_loss_coef  # pose_param=root_rel のときだけ使う
        self.task = 'pose' if 'keypoints' in losses else 'bbox'
        # 箱の損失に GIoU 項を足すか（DETR 原実装は L1 + GIoU。本リポジトリは
        # L1 のみだった）。GT の水平サイズが 0.5 m 固定なので L1 は実質「中心の
        # L1」になり、勾配の大きさがずれ量によらず一定になる。IoU 系の項は
        # 近いほど勾配が強くなるため、mAP@0.75 が要求する最後の数 cm に効く。
        self.giou = giou and self.task == 'bbox'
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def _match_quality(self, outputs, targets, indices, idx):
        return match_quality(outputs, targets, indices, idx, self.task, self.quality_sigma)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.iou_aware and idx[0].numel() > 0:
            # マッチしたクエリの確率質量を「真クラス:q / ∅:1-q」に割り振った
            # soft target を作り、クラス重み付き CE をかける。q ≡ 1 のときは
            # 下の F.cross_entropy と厳密に一致する（正規化も含めて）。
            q = self._match_quality(outputs, targets, indices, idx).to(src_logits.dtype)
            target_dist = F.one_hot(target_classes, self.num_classes + 1).to(src_logits.dtype)
            target_dist[idx] = 0.0
            target_dist[idx[0], idx[1], target_classes_o] = q
            target_dist[idx[0], idx[1], self.num_classes] = 1.0 - q

            log_p = F.log_softmax(src_logits, dim=-1)
            w = self.empty_weight.to(src_logits.dtype)
            numer = -(target_dist * log_p * w).sum(-1).sum()
            denom = (target_dist * w).sum().clamp(min=1e-6)
            loss_ce = numer / denom
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx]
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if 'pred_root' in outputs:
            # root_rel: ルート（腰中点）と相対オフセットを別々に L1。GT のルートは
            # left_hip / right_hip（OUTPUT_JOINTS の 6, 7 番）の中点。
            tgt_root = target_keypoints[:, [6, 7]].mean(1)                      # (n, 3)
            tgt_offset = (target_keypoints - tgt_root.unsqueeze(1)).flatten(1)  # (n, 42)
            loss_root = F.l1_loss(outputs['pred_root'][idx], tgt_root, reduction='sum')
            loss_offset = F.l1_loss(outputs['pred_offset'][idx], tgt_offset, reduction='sum')
            return {'loss_keypoint': (loss_offset + self.root_loss_coef * loss_root) / num_boxes}
        target_keypoints = target_keypoints.flatten(1)
        loss_keypoint = F.l1_loss(src_keypoints, target_keypoints, reduction='none')
        return {'loss_keypoint': loss_keypoint.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        if self.giou:
            giou = generalized_box_iou_3d(box_cxcywhd_to_xyzxyz(src_boxes),
                                          box_cxcywhd_to_xyzxyz(target_boxes))
            losses['loss_giou'] = (1.0 - torch.diag(giou)).sum() / num_boxes

        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'keypoints': self.loss_keypoints,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    
SELECT_MODES = ('legacy', 'detr', 'flat_topk', 'flat_all')


def select_detections(pred_logits, mode='flat_all', topk=20):
    """DETR 出力 (B, Q, C+1) のロジットから、評価に載せる検出集合を作る。

    返すのは (scores, labels, query_idx, keep) の 4 つで、いずれも (B, K)。
    boxes / keypoints は query_idx で引き直してから keep で絞る。

    mode:
      legacy    現行実装。∅ 列を含めた argmax を取り、∅ になったクエリを捨てる。
                1 フレーム 1〜3 件に収まるので CSV 出力・可視化向き。ただし
                全クエリの約 8 割が捨てられるため mAP の recall 天井を下げる。
      detr      DETR 原実装 (facebookresearch/detr の PostProcess) と同じ。
                ∅ 列を除いた最大確率を score とし、クエリを間引かない。
      flat_topk (クエリ, クラス) の全組を別々の検出として並べ、score 上位 topk を出す。
                Deformable DETR / DINO の後処理。行動クラスに確率が割れたクエリを
                どちらか一方に切り捨てずに済む。
      flat_all  flat_topk の topk 無し（K = Q*C）。

    score に objectness (1 - p(∅)) を使う案も検討したが、本リポジトリの mAP は
    クラス別 AP なので「そのクラスである確信度」を落とす方向に働き、実測でも
    悪化した（legacy 0.1070 / detr 0.1178 に対し objectness 版 0.1120）ため採らない。
    """
    if mode not in SELECT_MODES:
        raise ValueError(f'unknown select mode: {mode} (choices: {SELECT_MODES})')

    prob = F.softmax(pred_logits, -1)
    bs, num_queries, num_ch = prob.shape
    fg = prob[..., :-1]                                   # 背景 ∅ 列を除いた前景確率
    device = prob.device

    if mode == 'legacy':
        scores, labels = prob.max(-1)
        keep = labels != (num_ch - 1)
        query_idx = torch.arange(num_queries, device=device).expand(bs, -1)
        return scores, labels, query_idx, keep

    if mode == 'detr':
        scores, labels = fg.max(-1)
        query_idx = torch.arange(num_queries, device=device).expand(bs, -1)
        return scores, labels, query_idx, torch.ones_like(labels, dtype=torch.bool)

    num_classes = fg.shape[-1]
    flat = fg.reshape(bs, num_queries * num_classes)
    labels = torch.arange(num_classes, device=device).repeat(num_queries).expand(bs, -1)
    query_idx = torch.arange(num_queries, device=device).repeat_interleave(num_classes).expand(bs, -1)

    if mode == 'flat_topk' and topk is not None and topk < flat.shape[1]:
        scores, sel = flat.topk(topk, dim=1)
        labels = torch.gather(labels, 1, sel)
        query_idx = torch.gather(query_idx, 1, sel)
    else:
        scores = flat

    return scores, labels, query_idx, torch.ones_like(labels, dtype=torch.bool)


class PostProcess(nn.Module):
    @torch.no_grad()
    def __init__(self, num_classes, task='bbox', select='legacy', topk=20):
        super().__init__()
        self.no_object_class_idx = num_classes
        self.task = task
        # ここの出力は予測 CSV（可視化・定性評価用）にしか使われない。mAP は
        # engine.py が args.ap_select で作り直すので、こちらは 1 フレーム数件に
        # 収まる legacy を既定にして CSV の肥大と描画崩れを避ける。
        self.select = select
        self.topk = topk
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes, target_origins=None):
        """
        正規化座標を実空間（メートル）に戻す。target_sizes / target_origins は
        dataset.py と同じ [h, w, d]（= Y, X, Z）順で受け取り、ここで [w, h, d] に組み替える。
        逆変換は v = norm * span + origin のアフィン。target_origins を省略すると
        従来どおり乗算だけになる（原点中心のデータでは正しくないので、評価経路では必ず渡すこと）。
        """
        out_logits = outputs['pred_logits']
        assert len(out_logits) == len(target_sizes)

        scores, labels, query_idx, keep = select_detections(out_logits, self.select, self.topk)
        img_h, img_w, img_d = target_sizes.unbind(1)
        if target_origins is not None:
            assert len(out_logits) == len(target_origins)
            org_h, org_w, org_d = target_origins.unbind(1)

        results = []

        if self.task == 'pose':
            out_keypoints = outputs['pred_keypoints']
            keypoints = out_keypoints.reshape(out_keypoints.shape[0], out_keypoints.shape[1], 14, 3)
            scale_fct = torch.stack([img_w, img_h, img_d], dim=1).view(-1, 1, 1, 3)
            keypoints = keypoints * scale_fct
            if target_origins is not None:
                origin_fct = torch.stack([org_w, org_h, org_d], dim=1).view(-1, 1, 1, 3)
                keypoints = keypoints + origin_fct
            for s, l, q, kpts, k in zip(scores, labels, query_idx, keypoints, keep):
                results.append({'scores': s[k], 'labels': l[k], 'keypoints': kpts[q][k]})

        else:
            out_bbox = outputs['pred_boxes']
            boxes = box_cxcywhd_to_xyzxyz(out_bbox)
            scale_fct = torch.stack([img_w, img_h, img_d, img_w, img_h, img_d], dim=1)
            boxes = boxes * scale_fct[:, None, :]
            if target_origins is not None:
                origin_fct = torch.stack([org_w, org_h, org_d, org_w, org_h, org_d], dim=1)
                boxes = boxes + origin_fct[:, None, :]
            for s, l, q, b, k in zip(scores, labels, query_idx, boxes, keep):
                results.append({'scores': s[k], 'labels': l[k], 'boxes': b[q][k]})

        return results
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def build(args):
    task = args.task
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETR(backbone, transformer, num_classes=num_classes, num_queries=args.num_queries, aux_loss=args.aux_loss, args=args)
    matcher = build_matcher(args)
    
    cls_coef = getattr(args, 'class_loss_coef', 1)

    if task == 'pose':
        weight_dict = {'loss_ce': cls_coef, 'loss_keypoint': args.keypoint_loss_coef}
        losses = ['labels', 'keypoints', 'cardinality']
    else:
        weight_dict = {'loss_ce': cls_coef, 'loss_bbox': args.bbox_loss_coef}
        if getattr(args, 'giou', True):
            weight_dict['loss_giou'] = args.giou_loss_coef
        losses = ['labels', 'boxes', 'cardinality']

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,
                             iou_aware=getattr(args, 'iou_aware', True),
                             quality_sigma=getattr(args, 'quality_sigma', 0.05),
                             giou=getattr(args, 'giou', True),
                             root_loss_coef=getattr(args, 'root_loss_coef', 14.0))
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_classes=num_classes, task=task)}

    return model, criterion, postprocessors