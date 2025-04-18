cd smoke

accelerate launch --config_file default_config.yaml \
--main_process_port 29700 \
--gpu_ids 0,1 \
train_2d.py \
--exp_id 'super_sim' \
--is_wavelet True \
--is_condition_control True \
--is_super_model True \
--N_downsample 2 \
--train_num_steps 300000 
