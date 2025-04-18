cd burgers

python eval_ddpm_burgers.py \
--dataset 1d \
--is_wavelet True \
--is_super_model True \
--upsample_t 3 \
--upsample_x 3 \
--exp_id 'base_sim' \
--checkpoint 8 \
--dim 128 \
--super_exp_id 'super_sim' \
--super_checkpoint 24 \
--super_dim 64 \
--Ntest 200 \
--batch_size 5 \
--using_ddim True \
--ddim_sampling_steps 50 \
--ddim_eta 1 \
--is_condition_u0 True \
--is_condition_f True \
--save_file results/evaluate/ddim_super_sim.yaml 