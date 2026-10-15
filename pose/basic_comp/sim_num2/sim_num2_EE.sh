# new_model3 (Illuminance_DETR) 用。DR/pose/4_action/EE.sh と同じ設定。
# 実行は new_model3/ ルートから: bash pose/4_action/EE.sh
#
# DR からの変更点(全スクリプト共通。詳細は new_model3/pose/README_CHANGES.txt 参照):
#   - --iou_aware は pose タスクなら自動解決される(iou_aware=False)。
#   - checkpoint選択・early stoppingの既定(--select_metric auto)がpose向けに
#     「-val MPJPE単独」に変更された(DRは -MPJPE/100+Acc+F1 の混合)。
#   - --set_cost_class / --set_cost_keypoint / --class_loss_coef はpose既定では
#     8 / 2.0 / 12 に自動解決されるが、学習曲線解析の結果、分類はどの重みでも
#     数epochで飽和し座標側の学習を圧迫するだけと分かったため、DR相当の
#     1 / 1 / 1 に明示的に戻している。

CUDA_VISIBLE_DEVICES=1 python -u Illuminance_DETR/main.py \
    --train 4_action_new_af/4_action_new_af_train  sim_new_4_action_100w sim_new_4_action_180w \
    --val   4_action_new_af/4_action_new_af_val \
    --test  4_action_new_af/4_action_new_af_test \
    --exp_name 4ac_EE \
    --task pose --num_classes 4 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment EE \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
