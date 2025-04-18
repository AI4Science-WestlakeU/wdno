cd burgers

python train_ddpm_burgers.py \
    --exp_id 'super_sim' \
    --is_wavelet True \
    --is_super_model True \
    --train_num_steps 250000 \
    --test_interval 10 \
    --is_condition_u0 True \
    --is_condition_f True \
    --dim 64 \
    --N_downsample 3 