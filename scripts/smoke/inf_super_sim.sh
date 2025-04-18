cd smoke

python inference_2d.py \
--is_super_model True \
--upsample 1 \
--exp_id "base_sim" \
--super_exp_id "super_sim" \
--is_wavelet True \
--diffusion_checkpoint 50 \
--super_diffusion_checkpoint 75 \
--is_condition_control True \
--batch_size 25 \
--using_ddim True \
--ddim_eta 1 \
--ddim_sampling_steps 100 