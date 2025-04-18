cd burgers

python ddpm_burgers/generate_burgers.py \
--end_time 8 \
--nt 320 \
--nx 480 \
--train_samples 40000 \
--test_samples 8000 \
--save_path "data/1d/"

python ddpm_burgers/generate_burgers.py \
--end_time 8 \
--nt 1280 \
--nx 1920 \
--train_samples 0 \
--test_samples 8000 \
--save_path "data/1d_super/"