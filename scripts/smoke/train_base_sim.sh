cd smoke

accelerate launch --config_file default_config.yaml \
--main_process_port 29500 \
--gpu_ids 0,1 \
train_2d.py \
--exp_id 'base_sim' \
--is_wavelet True \
--is_condition_control True 
