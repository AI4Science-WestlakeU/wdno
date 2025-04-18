cd smoke

python inference_2d.py \
--exp_id "base_control" \
--is_wavelet True \
--diffusion_checkpoint 50 \
--using_ddim True \
--ddim_eta 1 \
--ddim_sampling_steps 100 \
--standard_fixed_ratio_list 100 \
--w_init_list 0.1 