cd burgers

python train_ddpm_burgers.py \
    --exp_id 'base_control' \
    --dataset 1d \
    --is_wavelet True \
    --is_condition_u0 True \
    --is_condition_uT True 