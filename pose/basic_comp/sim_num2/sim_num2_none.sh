# new_model3 (Illuminance_DETR) 用。DR/pose/4_action/none.sh と同じ設定。
# 実行は new_model3/ ルートから: bash pose/4_action/none.sh
# 変更点は EE.sh の冒頭コメント、および pose/README_CHANGES.txt を参照。

CUDA_VISIBLE_DEVICES=1 python -u Illuminance_DETR/main.py \
    --train 4_action_new_af/4_action_new_af_train  sim_new_4_action_100w sim_new_4_action_180w \
    --val   4_action_new_af/4_action_new_af_val \
    --test  4_action_new_af/4_action_new_af_test \
    --exp_name 4ac_none \
    --task pose --num_classes 4 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment none \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
