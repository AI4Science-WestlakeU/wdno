cd burgers

python eval_ddpm_burgers.py \
--exp_id 'base_sim' \
--checkpoint 8 \
--dataset 1d \
--is_wavelet True \
--Ntest 8000 \
--batch_size 200 \
--using_ddim True \
--ddim_sampling_steps 50 \
--ddim_eta 1 \
--is_condition_u0 True \
--is_condition_f True \
--save_file results/evaluate/ddim_base_sim.yaml 