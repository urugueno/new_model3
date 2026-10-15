import argparse
import datetime
import json
import random
from pathlib import Path
import csv
import fnmatch
import os
import functools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from util.logger import log_results
from datasets.dataset import build as build_dataset
from engine import evaluate, train, uses_map_selection, uses_mpjpe_selection
from models.detr import build as build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector and pose estimator', add_help=False)
    
    parser.add_argument('--task', default='bbox', type=str, choices=['bbox', 'pose', 'id'],
                        help='id: 個人識別（identify.py）。pose と同じ入力・バックボーンで '
                             '被験者 ID を分類する（1 セッション = 1 被験者）。')
    parser.add_argument('--id_clip_sec', type=float, default=5.0,
                        help='--task id のクリップ単位評価の長さ（秒）。')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # mAP に載せる検出集合の作り方。既定は Deformable DETR / DINO と同じ
    # 「(クエリ, クラス) の全組を検出として出す」後処理。legacy は従来実装
    # （∅ が argmax のクエリを捨てる）で、過去の結果を再現したいときに使う。
    parser.add_argument('--ap_select', type=str, default='flat_all',
                        choices=['legacy', 'detr', 'flat_topk', 'flat_all'],
                        help='mAP 用の後処理。legacy: 従来（∅ argmax を破棄）。'
                             'detr: DETR 原実装（前景 max・全クエリ）。'
                             'flat_topk / flat_all: (クエリ, クラス) 全組を展開。')
    parser.add_argument('--ap_topk', type=int, default=20,
                        help='--ap_select flat_topk のときに残す検出数。')

    # 分類ターゲットを 1.0 ではなく「その予測の当たり具合 q」にする
    # (IoU-aware classification: Generalized Focal Loss / VarifocalNet)。
    # softmax の前景確率がそのまま位置精度の推定量になるので、mAP の score
    # 順位づけが意味を持つ。提案手法・全ベースラインに同じ条件で入る。
    # 既定は「bbox のみ ON、pose は OFF」（default=None を main() で解決する）。
    # pose は mAP を計算しないので score 順位が改善しても得るものが無い一方、
    # 「当たっていない予測の確信度を下げる」副作用で検出の取りこぼしだけが増える。
    # 実測でも pose/ours の検出ゼロフレームが 7.6% → 12.1% に悪化した。
    # さらに pose の品質 q = exp(-d^2/2σ^2) は学習初期に ≈ 0 になるため、
    # bbox（初期 IoU ≈ 0.1）より分類の冷え込みが深い。
    # 明示指定した場合はそちらを優先する。
    parser.add_argument('--iou_aware', action='store_true', default=None,
                        help='分類ターゲットを IoU(bbox) / OKS(pose) にする。'
                             '既定は bbox のみ ON、pose は OFF。')
    parser.add_argument('--no_iou_aware', dest='iou_aware', action='store_false',
                        help='従来どおり分類ターゲットを 1.0 に固定する。')
    # checkpoint 選択 / early stopping に使う指標。従来の (Acc+IoU+F1)/3 は
    # mAP を一切見ておらず、実測でも accuracy 項はスコアの変動にほぼ寄与して
    # いなかった（相関 -0.14〜+0.36）。さらに IoU-aware 分類を入れると
    # accuracy が下がるため、従来指標では改善を「悪化」と誤判定する。
    parser.add_argument('--select_metric', type=str, default='auto',
                        choices=['auto', 'legacy', 'map', 'mpjpe'],
                        help='auto: bbox は val mAP@0.5:0.95、pose は val MPJPE。'
                             'legacy: 従来指標。map: val mAP（bbox のみ）。'
                             'mpjpe: val MPJPE（pose のみ）。')

    parser.add_argument('--quality_sigma', type=float, default=0.05,
                        help='pose の品質 q = exp(-d^2/2σ^2) の σ（正規化空間）。'
                             '既定 0.05 は空間幅 3 m に対して 15 cm。bbox では未使用。')

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    # B-2: マッチングで分類が占める比率。コストは cost_coord * L1 + cost_class * (-prob)
    # で、分類項は最大 1.0 にしかならない。一方 L1 は出力次元の「合計」なので、
    # bbox(6次元) では幾何コスト 0.48 に対し分類が 67% を占めるのに、
    # pose(42次元) では幾何コスト 10.5 に対し 8.7% しか占めない。その結果 pose では
    # 割り当てがほぼ幾何だけで決まり、分類ヘッドは「自分が影響できない割り当て」から
    # 教師信号を受け取ることになる（クエリの役割が安定せず、実測で特化が起きていない）。
    # 既定は task 依存（default=None を main() で解決）: bbox 1 / pose 8。
    # bbox は現状 1 で分類が割り当ての 28.7%（GIoU 込み）を占めており、検出も
    # 過不足 +0.10〜+0.20 とほぼ完璧なので動かさない。pose は 1 のままだと 8.7% しかなく、
    # 割り当てがほぼ幾何だけで決まってしまう。8 にすると 43.3% になる。
    parser.add_argument('--set_cost_class', default=None, type=float,
                        help='Hungarian マッチングでの分類コストの係数。'
                             '既定は bbox 1 / pose 8。')
    # pose のマッチングコスト係数。既定 None のとき --keypoint_loss_coef を流用する
    # （従来の挙動と完全に一致）。bbox は --set_cost_bbox と --bbox_loss_coef が
    # 独立しているのに、pose だけ損失係数に固定されていたのを分離するためのもの。
    # pose のマッチングでの座標コスト。既定 2（default=None を main() で解決）。
    # 従来は --keypoint_loss_coef (=10) を流用していたため、損失の重みを変えずに
    # マッチングだけ調整することができなかった。bbox は --set_cost_bbox と
    # --bbox_loss_coef が独立しているので、それに合わせて分離する。
    # set_cost_class=8 と合わせると分類が割り当ての 79.2% を占める（bbox の 67.4% に近い）。
    parser.add_argument('--set_cost_keypoint', default=None, type=float,
                        help='pose のマッチングでの座標コストの係数。既定 2（pose のみ）。')
    parser.add_argument('--eos_coef', default=0.3, type=float)
    
    # for bbox
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    # 箱の損失を DETR 原実装と同じ L1 + GIoU にする。既定 ON、係数も原実装と同じ 2。
    # 本リポジトリは長らく L1 のみで、GT の水平サイズが 0.5 m 固定のため
    # 「勾配の大きさがずれ量によらず一定」になっていた。
    parser.add_argument('--giou', action='store_true', default=True,
                        help='箱の損失とマッチャのコストに 3D GIoU を足す。既定 ON。')
    parser.add_argument('--no_giou', dest='giou', action='store_false',
                        help='従来どおり L1 のみにする。')
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--remove_difficult', action='store_true')
    
    # for pose
    parser.add_argument('--keypoint_loss_coef', default=15, type=float)
    # 関節座標の出し方。abs は従来どおり 42 次元の絶対座標を直接 sigmoid で出す。
    # root_rel は「ルート（腰中点）3 次元 + ルートからの相対 42 次元」に分けて出し、
    # pred_keypoints = root + offset として返す（マッチャ・評価・CSV は abs と共通）。
    # 損失だけが root と offset に分かれる。
    parser.add_argument('--pose_param', type=str, default='abs', choices=['abs', 'root_rel'],
                        help='pose の出力パラメータ化。abs: 絶対座標（従来）。'
                             'root_rel: ルート + 相対オフセット。')
    # 相対オフセットの上限（メートル、全軸共通）。tanh × この値 / space_size で出す。
    # 処理済みデータの実測で |関節 - 腰中点| は Z 最大 1.37 m、X/Y は 99.9% が 0.5 m 以内。
    parser.add_argument('--pose_offset_range', type=float, default=1.5,
                        help='root_rel の相対オフセットの最大値（m）。')
    # root_rel でのルート L1 の重み（offset の L1 に対する比）。abs の L1 では
    # ルートのずれが 14 関節すべてに乗るので、14 にすると root にかかる勾配が abs と揃う。
    parser.add_argument('--root_loss_coef', type=float, default=14.0,
                        help='root_rel でのルート L1 の重み（既定 14 = 関節数）。')

    # B-1: 分類損失の重み。既定 1 は従来と同じ挙動。
    # 損失に占める分類の比率は bbox で 65.6%、pose で 5.8% と 10 倍以上の開きがある
    # （loss_keypoint は 42 次元の L1 の合計 × 10 で約 10.5、loss_ce は約 0.64）。
    # 同一モデル・同一データで task だけが違う条件で、bbox の検出は過不足 +0.1〜+0.2 と
    # ほぼ完璧なのに対し pose は 1 人シーンで 78.5% が検出ゼロになる。
    # pose を bbox 並み（60%台）にするなら 10〜16 が目安。
    # 既定は task 依存（default=None を main() で解決）: bbox 1 / pose 12。
    # bbox を 12 にすると分類が損失の 95.8% を占めて座標側の勾配がほぼ消えるので、
    # bbox は 1 のまま据え置く。pose は 1 だと 5.8%、12 で 45.1%（bbox は 65.6%）。
    parser.add_argument('--class_loss_coef', default=None, type=float,
                        help='分類損失 loss_ce の重み。既定は bbox 1 / pose 12。')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_runs', default=1, type=int)
    parser.add_argument('--early_stopping_patience', default=50, type=int)

    # 既定 0。現行の DataLoader は num_workers を渡しておらず暗黙 0 なので、
    # 既定 0 なら照度実行のスループット・乱数消費・メモリ挙動はすべて不変。
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--num_sensors', type=int, default=36)
    parser.add_argument('--sensor_dropout_rate', type=float, default=0.0)
    parser.add_argument('--random_sensors', action='store_true')
    
    parser.add_argument('--window_size', type=int, default=41)
    parser.add_argument('--sub_window_size', type=int, default=37)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--k_neighbors', type=int, default=36)

    parser.add_argument('--data', default=None, nargs='+', type=str)
    parser.add_argument('--train', default=None, nargs='+', type=str)
    parser.add_argument('--val', default=None, nargs='+', type=str)
    parser.add_argument('--test', default=None, nargs='+', type=str)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--split_file', type=str, default=None,
                        help='セッション単位の分割を定義した JSON。--data / --train とは排他。'
                             'pool と各 fold の test / val を持ち、train は pool から両者を除いて導出する。')
    parser.add_argument('--split_mode', type=str, default='session', choices=['session', 'within'],
                        help='session: セッション単位のヘルドアウト（従来）。'
                             'within: 指定セッションすべてを時系列で --train_ratio / --val_ratio に分割する。'
                             '--split_file 指定時は pool 全体、--train 指定時は train+val+test を使う。')
    parser.add_argument('--folds', type=str, nargs='+', default=None,
                        help='--split_file のうち実行する fold 名（例: A B）。省略時はファイル内の全 fold。')
    parser.add_argument('--fold_name', type=str, default=None, help=argparse.SUPPRESS)
    
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    parser.add_argument('--standardize', action='store_false')
    parser.add_argument('--standardize_threshold', default=5.0, type=float)
    # 正規化空間 [0,1] と実空間（メートル）の対応をここだけで定義する。
    #   norm = (v - space_origin) / space_size,  v = norm * space_size + space_origin
    # pose.csv / bbox_label.csv はメートル・原点中心（real は足首の広がりを area_size=2.0
    # にリスケール、sim は部屋中心原点）なので、割るだけでなく必ずシフトも要る。モデルの座標
    # 出力は sigmoid で (0,1) しか取れないため、目標値が [0,1] に収まらないと到達不能になる。
    # 既定値の根拠: real 全 39 セッションの pose.csv（2.09e7 座標）で、X/Y ±1.5 m・
    # Z [-0.5, 2.5] m の枠から外れるのは 0.0018%（370 座標）だけ。±1.0 = area_size/2 だと
    # 腕が床面エリアの外に出るため 2.78% が範囲外になる。枠外は dataset.py が clamp し、
    # 件数を警告する。
    # 上の枠外率は足首基準（preprocess_real.py --rescale_ref ankle）の座標での実測。
    # 腰基準だった修正前は X/Y が約 1.13 倍で、同じ枠で 0.0024%、±1.0 では 8.15% だった。
    parser.add_argument('--space_size', nargs=3, type=float, default=[3.0, 3.0, 3.0],
                        help='正規化空間 [0,1] に対応する実空間の幅 (X, Y, Z / メートル)。')
    parser.add_argument('--space_origin', nargs=3, type=float, default=[-1.5, -1.5, -0.5],
                        help='正規化空間の 0 に対応する実座標 (X, Y, Z / メートル)。')
    parser.add_argument('--metric_scale', type=float, default=1000.0,
                        help='MPJPE / mpjdle をログに出すときの m からの換算係数。既定 1000 で mm。')


    parser.add_argument('--model_mode', type=str, default='time_sensor',
                        choices=['time', 'time_sensor', 'time_sensor_no_time_pe',
                                 'time_sensor_no_sensor_pe', 'time_sensor_no_pe',
                                 'time_sensor_index_sine', 'time_sensor_index_learned',
                                 'time_sensor_coord_linear', 'time_sensor_coord_mlp'],
                        help='空間・時間位置符号化の設計アブレーション用。'
                             'time: センサ軸を使わず時間トークンのみ。 '
                             'time_sensor: 座標の固定正弦波PE（提案手法/SCE）+時間PE。 '
                             'time_sensor_no_time_pe: time_sensorから時間PEのみ除いたもの。 '
                             'time_sensor_no_sensor_pe: 空間PEなし（時間PEのみ）。 '
                             'time_sensor_no_pe: 空間・時間PEともになし。 '
                             'time_sensor_index_sine: センサスロット番号に固定正弦波を適用+時間PE。 '
                             'time_sensor_index_learned: センサIDごとの学習可能埋め込み+時間PE（旧wifi_like）。 '
                             'time_sensor_coord_linear: 座標(x,y)を線形層1枚で符号化+時間PE。 '
                             'time_sensor_coord_mlp: 座標(x,y)を学習可能なMLPで符号化+時間PE。')
    parser.add_argument('--scale', type=str, default='full', 
                        choices=['full', 'reductionCNNconfidence', 'reductionVariance', 'reductionAttention'])
    parser.add_argument('--environment', type=str, default='none', choices=['none', 'EE', 'EQ'])
    parser.add_argument('--adapt', type=str, default='none', choices=['none', 'adabn', 'input', 'batch'],
                        help='none: 通常通り。adabn: BatchNormのrunning statsをtestデータ(ラベル不使用)で再計算してから最終test評価を行う。input: 入力標準化(mean/std)をtestデータ自身の統計量(ラベル不使用)で計算し直す。batch: 評価時、各バッチをそれ以前の過去バッチの累積統計量(ラベル不使用)でオンライン正規化する(1バッチ目のみtrain統計量を使用)。')
    parser.add_argument('--adapt_ratio', type=float, default=0.2,
                        help='--adapt input使用時、各testセッションの先頭何割を正規化統計量の計算に使うか（残りを評価に使用。両者は排他）。')
    parser.add_argument('--adapt_momentum', type=float, default=0.1,
                        help='--adapt batch使用時の指数移動平均の重み（新しいバッチの寄与度）。PyTorchのBatchNorm momentumと同じ意味。')

    # ベースライン手法（'none' の場合は提案手法 models/detr.py を使う）
    parser.add_argument('--baseline', type=str, default='none',
                        choices=['none', 'person_in_wifi_3d', 'intelligent_carpet',
                                 'simple_baseline_integral', 'listening_human', 'x_fi',
                                 'inverse_super_resolution', 'ligest', 'soundmhpe', 'awc'],
                        help='比較手法を使う場合に指定する。実装は baseline/<name>/ 配下。')
    # ------------------------------------------------------------------
    # モダリティ切替。'none' なら従来どおり照度入力（既存の挙動は一切変わらない）。
    # 実装は modality/<name>/ 配下。ベースライン（baseline/）が「照度入力のまま手法を
    # 差し替える」のに対し、こちらは入力モダリティそのものが変わるので Dataset と
    # collate も差し替わる。
    parser.add_argument('--modality', type=str, default='none', choices=['none', 'rgb', 'sound'],
                        help="他モダリティとの直接比較用。'rgb' で単眼RGB画像入力、"
                             "'sound' で 4ch アンビソニックス（可聴 TSP のアクティブ音響センシング）入力になる。")
    parser.add_argument('--rgb_model', type=str, default='mppe', choices=['mppe', 'integral'],
                        help='mppe: 3DMPPE (RootNet+PoseNet, ICCV 2019)。'
                             'integral: Integral Human Pose Regression (ECCV 2018)。')
    parser.add_argument('--rgb_det_cache', type=str, default='det_person_frcnn.csv',
                        help='人物検出のキャッシュ CSV 名（processed/<session>/ 配下）。'
                             'modality/rgb/tools/cache_detections.py で作る。')
    parser.add_argument('--rgb_det_score', type=float, default=0.3,
                        help='採用する検出スコアの閾値。全件が閾値未満のフレームでは'
                             '閾値を無視して上位を使うので、低めでも取りこぼさない。')
    parser.add_argument('--rgb_max_person', type=int, default=4,
                        help='1 フレームあたりのクロップ数 P。--num_queries と一致させること'
                             '（RGB ではクエリ 1 つが検出クロップ 1 つに対応する）。'
                             'GT の最大人数は 3 なので 4 で背景スロット 1 個ぶんの余裕がある。')
    parser.add_argument('--rgb_affine_json', type=str, default=None,
                        help='カメラ座標→部屋座標のアフィン係数 JSON。'
                             'modality/rgb/tools/fit_affine.py で作る。')
    parser.add_argument('--rgb_undistort', action='store_true', default=True,
                        help='逆投影で歪み補正を行う（既定 ON）。GT 生成に使われた'
                             'librealsense と同じ反復を通すので厳密に一致する。')
    parser.add_argument('--rgb_no_undistort', dest='rgb_undistort', action='store_false')
    parser.add_argument('--rgb_frame_stride', type=int, default=1,
                        help='train のみフレームを間引く。val/test には適用しない'
                             '（評価対象が照度実行とずれて比較が無効になるため）。')
    parser.add_argument('--rgb_aug_scale', type=float, default=0.25,
                        help='ボックスのスケール拡張の幅。0 で無効。')
    parser.add_argument('--rgb_aug_shift', type=float, default=0.10,
                        help='ボックスの平行移動拡張の幅（辺長に対する割合）。')
    parser.add_argument('--rgb_aug_color', type=float, default=0.20,
                        help='色ジッタ（チャネルごとのゲイン）の幅。')
    parser.add_argument('--rgb_amp', action='store_true', default=True,
                        help='bf16 autocast を使う（既定 ON）。bf16 は指数部が fp32 と'
                             '同じなので GradScaler が要らず、main.py 側に AMP の'
                             'プラミングを入れずに済む。')
    parser.add_argument('--rgb_no_amp', dest='rgb_amp', action='store_false')
    # ---- 音モダリティ（--modality sound）。前処理は preprocessing/sound/ 参照 ----
    parser.add_argument('--sound_model', type=str, default='cnn',
                        choices=['cnn', 'token', 'smhpe'],
                        help='cnn: 時間xメルの2次元マップを小型CNNで畳む（既定）。ヘッドは照度の'
                             '提案手法と共通なので「ヘッドを固定して入力だけ替えた」比較になる。'
                             'token: 照度側 --model_mode time と同じトークン化のアブレーション。'
                             'smhpe: 本家 SoundMHPE (ECCV 2026) の再現。モデル・損失・マッチャは'
                             ' baseline/soundmhpe/ と共通なので --baseline soundmhpe（照度入力）'
                             'と並べると「手法を固定して入力だけ替えた」比較になる。'
                             '前処理を --preset soundmhpe で作っておくこと。')
    parser.add_argument('--sound_token_mode', type=str, default='mic', choices=['mic', 'chunk'],
                        help='--sound_model smhpe のトークン化。mic: (時間フレーム, チャンネル)'
                             'が 1 トークン（原実装の (時間, マイク) トークンに対応。既定）。'
                             'chunk: チャンネルを特徴ベクトルに畳んで時間フレームだけをトークンにする。')
    parser.add_argument('--sound_logmel_only', action='store_true', default=False,
                        help='音響インテンシティ 3ch を捨てて log-mel 4ch だけを使う。'
                             '本家の公開コードの既定経路（libs/dataset.py の transform_logmel）'
                             'がこれに当たる。論文本文は 7ch なので、既定は 7ch のまま。')
    parser.add_argument('--sound_window', type=int, default=16,
                        help='入力に使う音フレーム数（30 fps）。既定 16 = 0.53 秒で、'
                             '照度の --window_size 41 行 @約78Hz と同じ情報地平になる。'
                             'トークン数を揃えたいときは 41 を指定する。')
    parser.add_argument('--sound_channels', type=int, default=7,
                        help='特徴チャンネル数。log-mel 4ch(W,X,Y,Z) + 音響インテンシティ 3ch。')
    parser.add_argument('--sound_n_mels', type=int, default=128,
                        help='メル帯域数。前処理の --n-mels と一致させること（起動時に検査する）。')
    parser.add_argument('--sound_cnn_width', type=int, default=64,
                        help='--sound_model cnn の 1 段目のチャンネル数（以降 1,2,4,4 倍）。')
    parser.add_argument('--sound_cnn_strides', type=str, default='1x2,2x2,2x2,1x2',
                        help='段ごとの (時間)x(メル) ストライド。既定は (16,128) -> (4,4) = 16 トークン。')
    parser.add_argument('--sound_sub_window', type=int, default=8,
                        help='--sound_model token のスライス長（照度の --sub_window_size に相当）。')
    parser.add_argument('--sound_stride', type=int, default=1,
                        help='--sound_model token のスライス間隔（照度の --stride に相当）。')
    parser.add_argument('--sound_allow_missing', dest='sound_require_full',
                        action='store_false', default=True,
                        help='音の特徴が無いフレームを除外して続行する。既定は停止する'
                             '（黙って評価フレーム集合が照度実行とずれるのを防ぐため）。')
    # 3DMPPE
    parser.add_argument('--mppe_backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--mppe_input_size', type=int, default=256)
    parser.add_argument('--mppe_depth_dim', type=int, default=64,
                        help='ボリュメトリックヒートマップの深さ方向の分割数。'
                             'OOM 時は 32 に落とす（soft-argmax が連続化するので実害は小さい）。')
    parser.add_argument('--mppe_num_deconv', type=int, default=3)
    parser.add_argument('--mppe_deconv_filters', type=int, default=256)
    parser.add_argument('--mppe_bbox_real', type=float, default=2000.0,
                        help='RootNet の k 値で仮定する人物の実寸 [mm]。原論文と同じ 2000。')
    parser.add_argument('--mppe_depth_range', type=float, default=1000.0,
                        help='root 相対深度の片側レンジ [mm]。原論文の bbox_3d_shape に対応。')
    parser.add_argument('--mppe_no_pretrained', dest='mppe_pretrained', action='store_false',
                        default=True,
                        help='trunk (ResNet) を ImageNet 事前学習なしのランダム初期化にする。'
                             '照度・音はゼロから学習するので、その条件にそろえる対照実験用。')
    parser.add_argument('--mppe_share_backbone', action='store_true',
                        help='RootNet と PoseNet で trunk を共有する（省メモリ）。')
    parser.add_argument('--mppe_learn_affine', action='store_true',
                        help='カメラ→部屋のアフィン 6 パラメータを学習する。train のみから'
                             '最適化されるのでリークは無いが、幾何経路のバグを吸収して'
                             '隠すので既定は凍結。')
    # ------------------------------------------------------------------
    # Person-in-WiFi 3D (PETR) の既定値。原論文 configs/wifi/petr_wifi.py に合わせる。
    parser.add_argument('--pw3d_enc_layers', type=int, default=6)
    parser.add_argument('--pw3d_dec_layers', type=int, default=3)
    parser.add_argument('--pw3d_refine_layers', type=int, default=2,
                        help='joint refine デコーダの層数。0 で無効化（OOM 時の退避用）。')
    parser.add_argument('--pw3d_dim_feedforward', type=int, default=1024)
    parser.add_argument('--pw3d_cls_weight', type=float, default=4.0)
    parser.add_argument('--pw3d_kpt_weight', type=float, default=70.0)
    # Intelligent Carpet: 疎な照度信号を IDW 補間で密なグリッドに変換してから CNN に通す。
    parser.add_argument('--carpet_grid_size', type=int, default=16,
                        help='IDW 補間先グリッドの一辺（圧力マップ相当の解像度）。')
    parser.add_argument('--carpet_idw_power', type=float, default=2.0,
                        help='逆距離加重の指数。大きいほど近傍センサを強く重視する。')
    # SimpleBaseline + Integral: ヒートマップ + soft-argmax。
    parser.add_argument('--sbi_grid_size', type=int, default=16,
                        help='IDW 補間先グリッドの一辺。')
    parser.add_argument('--sbi_idw_power', type=float, default=2.0)
    parser.add_argument('--sbi_depth_dim', type=int, default=8,
                        help='ボリュメトリックヒートマップの深さ(Z)方向の分割数。')
    parser.add_argument('--sbi_num_deconv', type=int, default=1,
                        help='deconv 層数。1増やすとヒートマップが縦横2倍になり、'
                             'メモリは約4倍になる（クエリ×関節×深さ分のチャネルを持つため）。')
    parser.add_argument('--sbi_deconv_filters', type=int, default=256)
    # Inverse Super Resolution: サブセルずらしの複数位相で低解像度観測を作りアンサンブルする。
    parser.add_argument('--isr_num_transforms', type=int, default=4,
                        help='生成する低解像度観測の数（変換の個数）。1 にすると ISR 無効相当。')
    parser.add_argument('--isr_grid_size', type=int, default=16,
                        help='IDW 補間先グリッドの一辺。')
    parser.add_argument('--isr_idw_power', type=float, default=2.0)
    parser.add_argument('--isr_fixed_transforms', action='store_true',
                        help='指定すると変換（オフセット/スケール）を学習せず固定する。')
    # X-Fi: X-Fusion（プーリングでトークン数を固定する self-attn + cross-attn）。
    # 既定値は原実装 X_Fi.py の hard-code に合わせる。
    parser.add_argument('--xfi_dim', type=int, default=512)
    parser.add_argument('--xfi_tokens', type=int, default=32,
                        help='X-Fusion が保持する共有埋め込みのトークン数（原実装は32）。')
    parser.add_argument('--xfi_depth', type=int, default=4,
                        help='X-Fusion の反復回数（原実装の model_depth）。')
    parser.add_argument('--xfi_heads', type=int, default=8)
    parser.add_argument('--xfi_resize', type=str, default='64,32',
                        help='(センサ, 時間) の2次元マップをリサイズする先 "H,W"。'
                             'CSIブランチの Resize([136,32]) に相当し、センサ本数非依存にする。')
    # AWC (Qi et al., CVPR 2026): MetaFi++ 相当エンコーダ + 学習初期の FIM 重み付き重み制約。
    # alpha / window の既定値は原論文の最良値 (alpha_S=20k, K=20)。
    parser.add_argument('--awc_dim', type=int, default=256)
    parser.add_argument('--awc_depth', type=int, default=4,
                        help='エンコーダの Transformer 層数。')
    parser.add_argument('--awc_heads', type=int, default=8)
    parser.add_argument('--awc_resize', type=str, default='64,32',
                        help='(センサ, 時間) の2次元マップをリサイズする先 "H,W"（X-Fi と同じ）。')
    parser.add_argument('--awc_alpha', type=float, default=2e4,
                        help='AWC 損失の係数 alpha_S。原論文は MPJPE(mm) 損失での値なので要調整。')
    parser.add_argument('--awc_window', type=int, default=20,
                        help='AWC 損失を掛ける学習初期のエポック数 K。')
    parser.add_argument('--awc_fim_batches', type=int, default=0,
                        help='各エポック冒頭の FIM 計算に使うバッチ数の上限（0 なら全バッチ）。')
    # Listening Human Behavior (音響): 2Dエンコーダ + 1D時間方向U-Net。
    parser.add_argument('--lh_base_channels', type=int, default=64,
                        help='2Dエンコーダの基準チャネル数（原実装は64→128→256→256）。')
    parser.add_argument('--sbi_backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='SimpleBaseline 原論文の標準は resnet50。既定は計算量を抑えた resnet18。')
    # LiGest (環境光ジェスチャ認識): DWT ハード閾値デノイジング → DWT 係数（波形の
    # 形を保つ特徴）→ 係数の強さでセンサを並べ替える座標非依存のラスタライズ →
    # 線形射影(PCA相当)。同一モダリティの先行研究。
    parser.add_argument('--ligest_dwt_levels', type=int, default=3,
                        help='Haar DWT の分解レベル数（デノイジング・特徴抽出の両方で共用）。')
    parser.add_argument('--ligest_raster_size', type=int, default=16,
                        help='ラスタライズ後のセンサ軸の行数。係数ノルム降順に並べたセンサを'
                             'この数まで adaptive average pooling でマージする'
                             '（物理座標は使わない）。')
    parser.add_argument('--ligest_compressed_coeffs', type=int, default=16,
                        help='「画像圧縮」段で係数軸を間引く先の幅。')
    parser.add_argument('--ligest_feat_dim', type=int, default=50,
                        help='PCA相当の線形射影後の最終特徴次元。原論文と同じ既定値50。')
    # SoundMHPE (音響マルチパーソン3D姿勢推定): マルチスケール時間トークン +
    # スケール内/スケール横断の2種類のエンコーダ自己注意 + 人物×時間に分解したクエリ。
    # 既定値は原実装 libs/config.py / opera/petr_sound_config.py に合わせる。
    parser.add_argument('--smhpe_enc_layers', type=int, default=3)
    parser.add_argument('--smhpe_dec_layers', type=int, default=2)
    parser.add_argument('--smhpe_dim_feedforward', type=int, default=1024)
    parser.add_argument('--smhpe_groups', type=int, default=6,
                        help='最粗スケールの時間チャンク数 G。窓を (G, 2G, 4G) 個の'
                             'チャンクに分けて3スケールのトークンを作る（原実装は 6/12/24 フレーム）。'
                             '窓長は 4G の倍数まで先頭側にパディングされる。')
    parser.add_argument('--smhpe_seq_len', type=int, default=4,
                        help='人物あたりの姿勢トークン数 S（クラストークンを除く）。原実装は 8 だが、'
                             '本データのラベルは窓末尾の1フレームだけなので既定を縮小した。'
                             '予測に使うのは最終フレームのトークンのみ。')
    parser.add_argument('--smhpe_token_mode', type=str, default='chunk',
                        choices=['chunk', 'sensor'],
                        help="chunk: 1時間チャンク=1トークン（センサ軸は特徴に畳む）。トークン数 7G。"
                             "sensor: 原実装に忠実に (チャンク, センサ)=1トークン。トークン数 7G*N に"
                             "なるので --smhpe_groups 1 程度と小さい --batch_size が前提。")
    parser.add_argument('--smhpe_cost_position', type=float, default=1.0,
                        help='マッチングの床面(X,Y)位置 L1 コストの係数。')
    parser.add_argument('--smhpe_cost_class', type=float, default=1.0,
                        help='マッチングのスコアコストの係数（コストから引かれる）。')
    parser.add_argument('--smhpe_pose_weight', type=float, default=1.0,
                        help='座標 MSE の重み（原実装 ratio_pose）。')
    parser.add_argument('--smhpe_class_weight', type=float, default=0.5,
                        help='分類 BCE の重み（原実装 ratio_class）。')
    return parser

def load_splits(args):
    """分割定義 JSON を読み、各 fold の train を pool から導出して返す。"""
    with open(args.split_file) as f:
        spec = json.load(f)

    pool = spec['pool']
    folds = spec['folds']

    if args.folds:
        by_name = {fold['name']: fold for fold in folds}
        missing = [n for n in args.folds if n not in by_name]
        if missing:
            raise ValueError(f"{args.split_file}: fold {missing} が見つかりません "
                             f"(定義されているのは {list(by_name)})")
        folds = [by_name[n] for n in args.folds]

    for fold in folds:
        held = fold['test'] + fold['val']
        duplicated = sorted({s for s in held if held.count(s) > 1})
        if duplicated:
            raise ValueError(f"fold {fold['name']}: test と val でセッションが重複 {duplicated}")
        unknown = sorted(set(held) - set(pool))
        if unknown:
            raise ValueError(f"fold {fold['name']}: pool 外のセッション {unknown}")
        fold['train'] = [s for s in pool if s not in set(held)]
        if not fold['train']:
            raise ValueError(f"fold {fold['name']}: train が空です")

    return folds

def run(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 学習データ・出力先の共通ルート。各自の環境に合わせて書き換える。
    PRIMARY_NAS_PATH = "/nas/data_2/ueno1212/light"
    SECONDARY_NAS_PATH = ""

    if os.path.isdir(PRIMARY_NAS_PATH):
        nas_path = PRIMARY_NAS_PATH
    elif os.path.isdir(SECONDARY_NAS_PATH):
        nas_path = SECONDARY_NAS_PATH
    else:
        nas_path = "."
        print(f"WARNING: Neither primary nor secondary NAS paths found. Using default path: {nas_path}.")

    data_path = f"{nas_path}/data"

    task = args.task
    label_csv_name = "pose.csv" if task == "pose" else "bbox_label.csv"
    out_dir_prefix = "output/pose" if task == "pose" else "output/bbox"
    if args.identify:
        out_dir_prefix = "output/id"

    if args.train:
        print(f"train: {args.train}\nval  : {args.val}\ntest : {args.test}")
        args.do_split = False
        args.train_illuminance_path = [f'{data_path}/{name}/light.csv' for name in args.train]
        args.train_bbox_path = [f'{data_path}/{name}/{label_csv_name}' for name in args.train]
        args.train_environment_path = [f'{data_path}/{name}/environment.csv' for name in args.train]

        args.val_illuminance_path = [f'{data_path}/{name}/light.csv' for name in args.val]
        args.val_bbox_path = [f'{data_path}/{name}/{label_csv_name}' for name in args.val]
        args.val_environment_path = [f'{data_path}/{name}/environment.csv' for name in args.val]

        args.test_illuminance_path = [f'{data_path}/{name}/light.csv' for name in args.test]
        args.test_bbox_path = [f'{data_path}/{name}/{label_csv_name}' for name in args.test]
        args.test_environment_path = [f'{data_path}/{name}/environment.csv' for name in args.test]
        
    else:
        print(f"train/val/test: {args.data}")
        args.do_split = True
        data_ill_paths = [f'{data_path}/{name}/light.csv' for name in args.data]
        data_bbox_paths = [f'{data_path}/{name}/{label_csv_name}' for name in args.data]
        data_env_paths = [f'{data_path}/{name}/environment.csv' for name in args.data]
        
        args.train_illuminance_path = args.val_illuminance_path = args.test_illuminance_path = data_ill_paths
        args.train_bbox_path = args.val_bbox_path = args.test_bbox_path = data_bbox_paths
        args.train_environment_path = args.val_environment_path = args.test_environment_path = data_env_paths

    # 照度固有のタグ（scale/model_mode/k/センサ本数）は RGB では意味を持たないので、
    # モダリティを切り替えたときはそちらをタグにする。
    if args.modality != 'none':
        sub_model = args.rgb_model if args.modality == 'rgb' else args.sound_model
        output_dir = (f'{nas_path}/{out_dir_prefix}/{args.exp_name}/'
                      f'{args.modality}_{sub_model}/queries_{args.num_queries}')
    else:
        output_dir = f'{nas_path}/{out_dir_prefix}/{args.exp_name}/{args.scale}_{args.model_mode}/k_{getattr(args, "k_neighbors", "none")}/{args.num_sensors}sensors'
        if args.random_sensors: output_dir += '_random'
    # セッション内分割の結果がヘルドアウトの log.txt に混ざらないよう分ける。
    if args.split_mode == 'within':
        output_dir += '/within_session'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 照度センサ列ごとの mean/std。RGB では意味を持たない（画像は ImageNet 統計で
    # 正規化する）ので else 側に落として mean/std/test_mean/test_std を None にする。
    if args.standardize and args.modality == 'none':
        train_paths = args.train_illuminance_path
        df_list = []
        for p in train_paths:
            df = pd.read_csv(p)
            
            if args.do_split:
                num_train = int(len(df) * args.train_ratio)
                df = df.iloc[:num_train]
            
            df_list.append(df)
        
        full_train_df = pd.concat(df_list, ignore_index=True)
        sensor_columns = [col for col in full_train_df.columns if col != 'timestamp']
        sensor_df = full_train_df[sensor_columns]
        
        if args.standardize_threshold > 0:
            sensor_df = sensor_df[sensor_df >= args.standardize_threshold]
        
        mean_vals = sensor_df.mean()
        std_vals = sensor_df.std()
        
        args.mean = torch.tensor(mean_vals.values, dtype=torch.float32)
        args.std = torch.tensor(std_vals.values, dtype=torch.float32)

        if args.adapt == 'input' and not args.do_split:
            args.test_mean = []
            args.test_std = []
            for p in args.test_illuminance_path:
                df = pd.read_csv(p)
                num_adapt = int(len(df) * args.adapt_ratio)
                session_df = df.iloc[:num_adapt][sensor_columns]

                if args.standardize_threshold > 0:
                    session_df = session_df[session_df >= args.standardize_threshold]

                args.test_mean.append(torch.tensor(session_df.mean().values, dtype=torch.float32))
                args.test_std.append(torch.tensor(session_df.std().values, dtype=torch.float32))
        else:
            if args.adapt == 'input' and args.do_split:
                print("Note: --adapt input is disabled in --data (do_split=True) mode; using train statistics for test as usual.")
            args.test_mean = None
            args.test_std = None

    else:
        args.mean = None
        args.std = None
        args.test_mean = None
        args.test_std = None

    # RGB ではセンサ列が存在しない。空にしておけば actual_num_sensors = 0、
    # selected_columns = [] となり、以降の照度専用処理がすべて無害化される。
    all_sensor_columns = ([] if args.modality != 'none'
                          else pd.read_csv(args.train_illuminance_path[0], nrows=0).columns.tolist()[1:])
    
    if 0 < args.num_sensors < len(all_sensor_columns):
        args.actual_num_sensors = args.num_sensors
    else:
        args.actual_num_sensors = len(all_sensor_columns)

    if not args.random_sensors:
        print(f"Running in Identical Sensor Mode: {args.actual_num_sensors} / {len(all_sensor_columns)}")
        if args.actual_num_sensors < len(all_sensor_columns):
            args.selected_columns = random.sample(all_sensor_columns, args.actual_num_sensors)
        else:
            args.selected_columns = all_sensor_columns
    else:
        print(f"Running in Layout-Agnostic Sensor Mode: {args.actual_num_sensors} / {len(all_sensor_columns)}")

    device = torch.device(args.device)
    if args.modality != 'none':
        from modality import build_modality
        print(f"Using modality: {args.modality} "
              f"({args.rgb_model if args.modality == 'rgb' else args.sound_model})")
        model, criterion, postprocessors = build_modality(args)
    elif args.baseline == 'none':
        model, criterion, postprocessors = build_model(args)
    else:
        from baseline import build_baseline
        print(f"Using baseline: {args.baseline}")
        model, criterion, postprocessors = build_baseline(args)
    # --task id: 組んだ pose モデルの pose ヘッド直前の特徴に被験者分類ヘッドを付け替える。
    # 分割定義に subjects（セッション -> 被験者 ID）があれば被験者単位のラベル、
    # 無ければ 1 セッション = 1 被験者としてセッション番号をラベルにする。
    if args.identify:
        from identify import build_identity
        # 時系列分割なら pool、セッション分割なら train -> val -> test の順に並べた和集合。
        args.id_sessions = (list(args.data) if args.do_split
                            else list(dict.fromkeys(args.train + args.val + args.test)))
        subject_map = args.id_subject_map or {}
        missing = [s for s in args.id_sessions if subject_map and s not in subject_map]
        if missing:
            raise SystemExit(f'--task id: 分割定義の subjects にセッション {missing} がありません。')
        session_subject = {s: subject_map.get(s, s) for s in args.id_sessions}
        args.id_class_names = list(dict.fromkeys(session_subject.values()))
        args.id_subject_index = {s: args.id_class_names.index(c) for s, c in session_subject.items()}
        args.id_num_subjects = len(args.id_class_names)
        model, criterion = build_identity(model, args)
        postprocessors = None
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    confidence_cnn_params = [p for n, p in model_without_ddp.named_parameters() 
                             if "confidence_cnn" in n and p.requires_grad]
    backbone_params = [p for n, p in model_without_ddp.named_parameters() 
                       if "backbone" in n and "reduction_cnn" not in n and "confidence_cnn" not in n and p.requires_grad]
    base_params = [p for n, p in model_without_ddp.named_parameters() 
                   if "backbone" not in n and "eq_cnn" not in n and p.requires_grad]
    eq_cnn_params = [p for n, p in model_without_ddp.named_parameters()
                                if "eq_cnn" in n and p.requires_grad]

    cnn_lr = args.lr * 10
    param_dicts = [
        {"params": base_params},
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": confidence_cnn_params, "lr": cnn_lr},
        {"params": eq_cnn_params, "lr": cnn_lr},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # モダリティごとに Dataset を差し替える。if/else ブロックにせずシンボルだけ
    # 差し替えることで、既存行の再インデントを避けている。
    _build_ds = build_dataset
    if args.modality != 'none':
        from modality import build_dataset as _build_ds

    dataset_train = _build_ds(image_set='train', args=args)
    dataset_val = _build_ds(image_set='val', args=args)

    if args.test_mean is not None:
        saved_mean, saved_std = args.mean, args.std
        args.mean, args.std = args.test_mean, args.test_std
        dataset_test = _build_ds(image_set='test', args=args)
        args.mean, args.std = saved_mean, saved_std
    elif args.adapt == 'batch':
        saved_mean, saved_std = args.mean, args.std
        args.mean, args.std = None, None
        dataset_test = _build_ds(image_set='test', args=args)
        args.mean, args.std = saved_mean, saved_std
    else:
        dataset_test = _build_ds(image_set='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    
    _collate = utils.collate_fn
    _worker_init = None
    if args.modality != 'none':
        from modality import get_collate_fn, get_worker_init_fn
        _collate = get_collate_fn(args)
        _worker_init = get_worker_init_fn(args)

    collate_train = functools.partial(_collate, args=args, split='train')
    collate_val   = functools.partial(_collate, args=args, split='val')
    collate_test  = functools.partial(_collate, args=args, split='test')

    # num_workers の既定は 0 なので、指定しない限り照度実行の挙動は変わらない。
    _dl_kw = dict(num_workers=args.num_workers,
                  pin_memory=args.num_workers > 0,
                  persistent_workers=args.num_workers > 0,
                  worker_init_fn=_worker_init if args.num_workers > 0 else None)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_train, **_dl_kw)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_val, **_dl_kw)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, collate_fn=collate_test, **_dl_kw)

    # --task id は学習・評価ループだけ identify.py のものに差し替える（_build_ds と同じ流儀）。
    _train, _evaluate = train, evaluate
    if args.identify:
        from identify import train_id as _train, evaluate_id as _evaluate

    if args.eval:
        _evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir, args=args)
        return

    # -MPJPE（mm）は序盤に -1000 を下回りうるので -inf から始める
    # --task id は (val 正解率, -val 損失) の辞書式比較。正解率が 1.0 で頭打ちしても損失で選べる。
    best_metric_score = (float('-inf'), float('-inf')) if args.identify else float('-inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    best_checkpoint_path = f"{output_dir}/best_{run_timestamp}.pth"

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = _train(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args=args)
        lr_scheduler.step()

        val_stats = _evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir, prediction_filename=f'val_prediction_latest.csv', header_prefix="Val", args=args)

        if args.identify or ('accuracy' in val_stats and 'f1_score' in val_stats):
            current_acc = val_stats.get('accuracy', 0.0)
            current_f1 = val_stats.get('f1_score', 0.0)

            if args.identify:
                current_score = (val_stats['id_acc'], -val_stats['loss'])
                print(f"Epoch {epoch}: Score (val id_acc, -loss): ({current_score[0]:.4f}, {current_score[1]:.4f}) "
                      f"(Best: ({best_metric_score[0]:.4f}, {best_metric_score[1]:.4f}))")
            elif uses_mpjpe_selection(args):
                # MPJPE は小さいほど良いので符号を反転してスコアにする（acc / f1 は見ない）
                current_mpjpe = val_stats.get('mpjpe', float('nan'))
                current_score = -current_mpjpe
                print(f"Epoch {epoch}: Score (-val MPJPE): {current_score:.4f} (Best: {best_metric_score:.4f})"
                      f"  [acc {current_acc:.4f} f1 {current_f1:.4f}]")
            elif task == 'pose':
                current_mpjpe = val_stats.get('mpjpe', 0.0)
                current_score = -current_mpjpe/100.0 + current_acc + current_f1
                print(f"Epoch {epoch}: Score (-MPJPE/100+Acc+F1): {current_score:.4f} (Best: {best_metric_score:.4f})")
            elif uses_map_selection(args):
                current_score = val_stats.get('ap', 0.0)
                current_iou = val_stats.get('iou', 0.0)
                print(f"Epoch {epoch}: Score (val mAP@0.5:0.95): {current_score:.4f} "
                      f"(Best: {best_metric_score:.4f})  [iou {current_iou:.4f} acc {current_acc:.4f} f1 {current_f1:.4f}]")
            else:
                current_iou = val_stats.get('iou', 0.0)
                current_score = (current_acc + current_iou + current_f1) / 3.0
                print(f"Epoch {epoch}: Score (Acc+IoU+F1)/3: {current_score:.4f} (Best: {best_metric_score:.4f})")

            if current_score > best_metric_score:
                print("  New best model found! Saving checkpoint.")
                best_metric_score = current_score
                patience_counter = 0
                torch.save({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args}, best_checkpoint_path)
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        print(f"Now Train: {output_dir}\n\n"+"-"*30+"\n")
        
        log_stats = {**{f'tr_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': v for k, v in val_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        base_metrics = ['mpjpe', 'mpjdle_x', 'mpjdle_y', 'mpjdle_z', 'f1_score', 'accuracy'] if task == 'pose' else ['ap', 'iou', 'f1_score', 'accuracy']
        if args.identify:
            base_metrics = ['loss', 'id_acc']
        fieldnames = ['epoch'] + [f'tr_{m}' for m in base_metrics] + [f'val_{m}' for m in base_metrics]
        remaining_keys = sorted([k for k in log_stats.keys() if k not in fieldnames])
        fieldnames.extend(remaining_keys)

        with open(f"{output_dir}/log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if epoch == args.start_epoch: writer.writeheader()
            writer.writerow(log_stats)

        EXCLUDE_ARGS = {"mean", "std", "actual_num_sensors", "*illuminance_path*", "*bbox_path*", "*environment_path*", "t_runs"}
        with open(f"{output_dir}/log.txt", "a") as f:
            if epoch == args.start_epoch:
                f.write("\n" + "="*80 + "\n")
                f.write(f"START TIME: {run_timestamp}\n")
                fold_suffix = (f"  (fold: {args.fold_name} @ {Path(args.split_file).name})"
                               if args.fold_name else "")
                f.write(f"RUN       : {args.t_runs} / {args.n_runs}{fold_suffix}\n")
                f.write("--- ARGS ---\n")
                filtered_args = [
                    (k, v)
                    for k, v in vars(args).items()
                    if not any(fnmatch.fnmatch(k, pattern) for pattern in EXCLUDE_ARGS)
                ]
                max_len = max([len(k) for k, _ in filtered_args], default=0)
                for k, v in filtered_args:
                    f.write(f"  {k:<{max_len}} : {v}\n")
                f.write("\n")
                f.write("--- DATASETS ---\n")
                if args.do_split:
                    f.write(f" Train:Val:Test = {args.train_ratio:.2f}:{args.val_ratio:.2f}:{1 - args.train_ratio - args.val_ratio:.2f}\n")
                    f.write(f" Source Dataset: {args.data}\n")
                else:
                    f.write(f" Train: {args.train}\n")
                    f.write(f" Val  : {args.val}\n")
                    f.write(f" Test : {args.test}\n")
                f.write("\n")
                f.write("--- SENSOR CONFIG ---\n")
                train_sensors = dataset_train.selected_columns
                f.write(f"[train] Using {len(train_sensors)} sensors:\n")
                for i in range(0, len(train_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in train_sensors[i:i+3]) + "\n")
                f.write("\n")
                val_sensors = dataset_val.selected_columns
                f.write(f"[val] Using {len(val_sensors)} sensors:\n")
                for i in range(0, len(val_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in val_sensors[i:i+3]) + "\n")
                f.write("\n")
                test_sensors = dataset_test.selected_columns
                f.write(f"[test] Using {len(test_sensors)} sensors:\n")
                for i in range(0, len(test_sensors), 3):
                    f.write("  " + "  ".join(f"{col:<20}" for col in test_sensors[i:i+3]) + "\n")
                f.write("\n")
                header = "".join([f"{key:<18}" for key in fieldnames])
                f.write(header + "\n")

            row_str = ""
            for key in fieldnames:
                value = log_stats.get(key)
                if isinstance(value, float):
                    row_str += f"{value:<18.5f}"
                else:
                    row_str += f"{str(value):<18}"
            f.write(row_str + "\n")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.\n")
            break

    final_log_stats = None
    if Path(best_checkpoint_path).exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        test_loader_for_eval = data_loader_test

        if args.adapt == 'adabn':
            adapted = utils.recompute_batchnorm_stats(model, data_loader_test, device)
            if not adapted:
                print("Warning: --adapt adabn was specified but no BatchNorm layers were found in the model (check --scale/--baseline).")
        elif args.adapt == 'batch':
            test_loader_for_eval = utils.OnlineNormalizingLoader(data_loader_test, dataset_train.mean, dataset_train.std, momentum=args.adapt_momentum)

        test_stats = _evaluate(model, criterion, postprocessors, test_loader_for_eval, device, output_dir, prediction_filename=f'test_prediction_{run_timestamp}.csv', header_prefix="Test", args=args)
        final_log_stats = {f'{k}': v for k, v in test_stats.items()}

        desired_key_order = ['mpjpe', 'mpjdle_x', 'mpjdle_y', 'mpjdle_z', 'f1_score', 'accuracy', 'loss_ce', 'loss_keypoints'] if task == 'pose' else ['ap', 'ap50', 'ap75', 'iou', 'f1_score', 'accuracy', 'loss_ce', 'loss_bbox']
        if args.identify:
            desired_key_order = ['id_acc', 'id_bacc', 'id_acc_clip', 'id_chance', 'loss']

        # 集計スクリプト（modality/tools/scatter_pose_id.py）用の機械可読な 1 行。
        # launch_id は main() 1 回ごとに共通なので、再実行しても最新の n_runs 本だけを拾える。
        with open(f"{output_dir}/test_results.jsonl", "a") as f:
            f.write(json.dumps({'launch_id': args.launch_id, 'run_timestamp': run_timestamp,
                                'seed': args.seed, 'fold': args.fold_name,
                                'task': 'id' if args.identify else task,
                                'metric_scale': args.metric_scale,
                                **{k: float(v) for k, v in final_log_stats.items()}}) + "\n")

        ordered_stats = [f"'{test_key}': {final_log_stats.get(test_key, 0):.4f}" for k in desired_key_order if (test_key := f'{k}') in final_log_stats]
        final_stats_str = ", ".join(ordered_stats)

        with open(f"{output_dir}/log.txt", "a") as f:
            f.write("\n--- TEST RESULTS ---\n")
            if args.fold_name:
                f.write("fold         :" + str(args.fold_name) + "\n")
            f.write(final_stats_str + "\n")
            f.write("run_timestamp:" + str(run_timestamp) + "\n")
            f.write("output_path  :" + str(output_dir) + "\n")
            f.write("\n")
        
    return final_log_stats, output_dir

def main(args):
    if args.split_file and (args.data or args.train):
        raise SystemExit('--split_file は --data / --train と併用できません。')
    args.launch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --task id: Dataset とバックボーンは pose とまったく同じものを組む（評価フレーム集合を
    # pose 実行と一致させるため）ので、内部では task='pose' として扱い、identify フラグで
    # 学習・評価ループと出力先だけを分岐する（identify.py 参照）。
    args.identify = (args.task == 'id')
    if args.identify:
        if args.modality == 'none' and args.baseline != 'none':
            raise SystemExit('--task id は照度では提案手法（--baseline none）のみ対応です。')
        if args.adapt != 'none':
            raise SystemExit('--task id では --adapt は使えません。')
        args.task = 'pose'

    # 指定が無いオプションは task で既定を決める。根拠は get_args_parser の
    # 該当箇所のコメントを参照。いずれも bbox 側は従来値のまま据え置き、
    # pose 側だけを「bbox と同じ重みバランスになる点」に寄せてある。
    #
    #   --iou_aware        bbox True  / pose False
    #   --class_loss_coef  bbox 1     / pose 12   （損失に占める分類: 65.6% / 45.1%）
    #   --set_cost_class   bbox 1     / pose 8    （割り当てに占める分類: 28.7% / 43.3%）
    #   --set_cost_keypoint          （pose のみ）2   （8 と合わせて 79.2%）
    #
    # pose 側の 12 / 8 / 2 は bbox との比率合わせから逆算した値で、学習による
    # 裏付けはまだ無い。従来値に戻すには --class_loss_coef 1 --set_cost_class 1
    # --set_cost_keypoint 10 を明示する。
    is_pose = (args.task == 'pose')
    if args.iou_aware is None:
        args.iou_aware = not is_pose
    if args.class_loss_coef is None:
        args.class_loss_coef = 12.0 if is_pose else 1.0
    if args.set_cost_class is None:
        args.set_cost_class = 8.0 if is_pose else 1.0
    if args.set_cost_keypoint is None and is_pose:
        args.set_cost_keypoint = 2.0
    cost_coord = (args.set_cost_keypoint if args.set_cost_keypoint is not None
                  else args.keypoint_loss_coef) if args.task == 'pose' else args.set_cost_bbox
    print(f"[config] task={args.task}  iou_aware={args.iou_aware}  "
          f"giou={getattr(args, 'giou', False) and args.task == 'bbox'}  "
          f"ap_select={getattr(args, 'ap_select', 'flat_all')}  "
          f"select_metric={'map' if uses_map_selection(args) else 'mpjpe' if uses_mpjpe_selection(args) else 'legacy'}")
    print(f"[config] loss  cls={args.class_loss_coef} : coord="
          f"{args.keypoint_loss_coef if args.task == 'pose' else args.bbox_loss_coef}"
          f"   match  cls={args.set_cost_class} : coord={cost_coord}")

    # 照度センサ前提のオプションは RGB では成立しない。黙って無視すると
    # 結果だけ静かに狂うので、入口で落とす。
    if args.modality != 'none':
        if args.task != 'pose':
            raise SystemExit(f'--modality {args.modality} は --task pose 専用です。')
        if args.adapt != 'none':
            raise SystemExit(f'--adapt は --modality {args.modality} では使えません'
                             '（照度の入力統計量を前提にした仕組みのため）。')
        if args.random_sensors or args.environment != 'none':
            raise SystemExit(f'--random_sensors / --environment は '
                             f'--modality {args.modality} では無効です。')
    if args.modality == 'rgb':
        if args.rgb_affine_json is None:
            raise SystemExit('--rgb_affine_json を指定してください。'
                             'modality/rgb/tools/fit_affine.py で作れます。')
    if args.modality == 'sound':
        if args.sound_window < 1:
            raise SystemExit('--sound_window は 1 以上にしてください。')
        if args.sound_model == 'token' and args.sound_sub_window > args.sound_window:
            raise SystemExit(f'--sound_sub_window ({args.sound_sub_window}) が '
                             f'--sound_window ({args.sound_window}) より長いです。')
        if args.sound_model == 'smhpe':
            if args.sound_window % 4 != 0:
                raise SystemExit(f'--sound_model smhpe では --sound_window を 4 の倍数に '
                                 f'してください（マルチスケールが 1:2:4 に割り切れません）。'
                                 f'現在 {args.sound_window}')
            if args.num_queries < 2:
                raise SystemExit('--sound_model smhpe では --num_queries を 2 以上に '
                                 'してください（人物間注意が成立しません）。')

    all_final_stats = []
    base_output_dir = None
    original_seed = args.seed

    # --task id の被験者対応（セッション -> 被験者 ID）。within で split_file を消す前に読む。
    args.id_subject_map = None
    if args.split_file:
        with open(args.split_file) as f:
            args.id_subject_map = json.load(f).get('subjects')

    # --split_mode within: 指定セッションをまとめて --data 扱いにし、
    # 既存の do_split（セッション内の時系列分割）経路に流す。
    if args.split_mode == 'within':
        if args.split_file:
            with open(args.split_file) as f:
                args.data = json.load(f)['pool']
            if args.folds:
                print(f"Note: --split_mode within では --folds {args.folds} を無視します。")
            args.split_file = None
        elif args.train:
            sessions = args.train + (args.val or []) + (args.test or [])
            args.data = list(dict.fromkeys(sessions))
            args.train = args.val = args.test = None

    # fold ごとに使用セッションを差し替える。--split_file なしなら従来どおり 1 通りのみ。
    folds = load_splits(args) if args.split_file else [None]

    for fold in folds:
        if fold is not None:
            args.train, args.val, args.test = fold['train'], fold['val'], fold['test']
            args.fold_name = fold['name']

        for i in range(args.n_runs):
            fold_prefix = f"Fold {args.fold_name} / " if args.fold_name else ""
            print(f"{'='*10} {fold_prefix}Run {i+1}/{args.n_runs} {'='*10}")
            args.seed = original_seed + i
            args.t_runs = i+1

            single_run_stats, output_dir = run(args)

            if single_run_stats:
                all_final_stats.append(single_run_stats)

            if base_output_dir is None: base_output_dir = output_dir

    if all_final_stats and base_output_dir:
        log_results(all_final_stats, base_output_dir, args)
    elif args.n_runs > 1:
        print(f"Could not calculate averages. Found {len(all_final_stats)} valid run(s).")

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
