CUDA_VISIBLE_DEVICES=0 python -u Illuminance_DETR/main.py \
    --train 4_action_new_af/4_action_new_af_train  sim_DR_ver2_processed/DR_002/DR_002_1 sim_DR_ver2_processed/DR_002/DR_002_2 sim_DR_ver2_processed/DR_002/DR_002_3 sim_DR_ver2_processed/DR_002/DR_002_4 sim_DR_ver2_processed/DR_006/DR_006_1 sim_DR_ver2_processed/DR_006/DR_006_2 sim_DR_ver2_processed/DR_006/DR_006_3 sim_DR_ver2_processed/DR_006/DR_006_4 sim_DR_ver2_processed/DR_010/DR_010_1 sim_DR_ver2_processed/DR_010/DR_010_2 sim_DR_ver2_processed/DR_010/DR_010_3 sim_DR_ver2_processed/DR_010/DR_010_4 sim_DR_ver2_processed/DR_014/DR_014_1 sim_DR_ver2_processed/DR_014/DR_014_2 sim_DR_ver2_processed/DR_014/DR_014_3 sim_DR_ver2_processed/DR_014/DR_014_4 \
    --val   4_action_new_af/4_action_new_af_val \
    --test  4_action_new_af/4_action_new_af_test \
    --exp_name pose_dr2_k4_r0 \
    --task pose --num_classes 4 --keypoint_loss_coef 1 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment none \
    --num_sensors 36 --k_neighbors 36 \
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
