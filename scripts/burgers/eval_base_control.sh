cd burgers

python eval_ddpm_burgers.py \
--exp_id 'base_control' \
--checkpoint 8 \
--dataset 1d \
--is_wavelet True \
--batch_size 25 \
--using_ddim True \
--ddim_sampling_steps 50 \
--ddim_eta 1 \
--is_condition_u0 True \
--is_condition_uT True \
--J_scheduler "cosine" \
--wus 120000 \
--wfs 0.00002 \
--save_file results/evaluate/ddim_base_control.yaml 