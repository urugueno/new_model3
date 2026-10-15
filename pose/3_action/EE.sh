# new_model3 (Illuminance_DETR) 用。DR/pose/3_action/EE.sh と同じ設定。
# 実行は new_model3/ ルートから: bash pose/3_action/EE.sh
# 変更点は pose/4_action/EE.sh の冒頭コメント、および pose/README_CHANGES.txt を参照。

CUDA_VISIBLE_DEVICES=1 python -u Illuminance_DETR/main.py \
    --train 3_action_new_af/3action_02_train  sim_3_action_af/loop1 \
    --val   3_action_new_af/3action_02_val \
    --test  3_action_new_af/3action_02_test \
    --exp_name 3ac_EE \
    --task pose --num_classes 3 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment EE \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 8 --set_cost_keypoint 2.0 --class_loss_coef 12
