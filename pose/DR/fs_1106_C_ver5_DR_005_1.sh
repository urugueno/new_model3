CUDA_VISIBLE_DEVICES=0 python -u Illuminance_DETR/main.py \
    --train 4_action_new_af/4_action_new_af_train  sim_fs_1106_C_D_af/C_ver5_DR_005_1 \
    --val   4_action_new_af/4_action_new_af_val \
    --test  4_action_new_af/4_action_new_af_test \
    --exp_name fs_1106_C_ver5_DR_005_1 \
    --task pose --num_classes 4 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 35 --n_runs 2 \
    --model_mode time_sensor --scale full --environment none \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
