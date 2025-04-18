cd smoke

python inference_2d.py \
--exp_id "base_sim" \
--is_wavelet True \
--diffusion_checkpoint 50 \
--is_condition_control True \
--batch_size 100 \
--using_ddim True \
--ddim_eta 1 \
--ddim_sampling_steps 100 