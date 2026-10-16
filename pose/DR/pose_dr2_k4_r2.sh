CUDA_VISIBLE_DEVICES=0 python -u Illuminance_DETR/main.py \
    --train 4_action_new_af/4_action_new_af_train  sim_DR_ver2_processed/DR_002/DR_002_1 sim_DR_ver2_processed/DR_002/DR_002_2 sim_DR_ver2_processed/DR_002/DR_002_3 sim_DR_ver2_processed/DR_002/DR_002_4 sim_DR_ver2_processed/DR_007/DR_007_1 sim_DR_ver2_processed/DR_007/DR_007_2 sim_DR_ver2_processed/DR_007/DR_007_3 sim_DR_ver2_processed/DR_007/DR_007_4 sim_DR_ver2_processed/DR_011/DR_011_1 sim_DR_ver2_processed/DR_011/DR_011_2 sim_DR_ver2_processed/DR_011/DR_011_3 sim_DR_ver2_processed/DR_011/DR_011_4 sim_DR_ver2_processed/DR_013/DR_013_1 sim_DR_ver2_processed/DR_013/DR_013_2 sim_DR_ver2_processed/DR_013/DR_013_3 sim_DR_ver2_processed/DR_013/DR_013_4 \
    --val   4_action_new_af/4_action_new_af_val \
    --test  4_action_new_af/4_action_new_af_test \
    --exp_name pose_dr2_k4_r2 \
    --task pose --num_classes 4 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment none \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
