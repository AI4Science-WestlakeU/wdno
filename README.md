# WDNO: Wavelet Diffusion Neural Operator (ICLR 2025)




## Environment
```code
bash env.sh
```

## Dataset
The datasets can be downloaded respectively in [link](https://drive.google.com/drive/folders/1W1tbQ7ltIDEQdHzUarFo9EYMe4ngxXcz).
Please place the corresponding datasets in the `data` folder under the experiment directory.

## Checkpoints
The checkpoints can be downloaded respectively in [link](https://drive.google.com/drive/folders/1W1tbQ7ltIDEQdHzUarFo9EYMe4ngxXcz).
Please place the corresponding checkpoints in the `results` folder under the experiment directory.

## 1D Burgers' Equation Simulation
Download with the link above or prepare data for WDNO:
```code
cd burgers
python wave_trans.py
```
training of Base-Resolution Model:
```code
bash /scripts/burgers/train_base_sim.sh
```
inference of Base-Resolution Model:
```code
bash /scripts/burgers/eval_base_sim.sh
```
training of Super-Resolution Model:
```code
bash /scripts/burgers/train_super_sim.sh
```
inference of Super-Resolution Model:
```code
bash /scripts/burgers/eval_super_sim.sh
```

## 2D Smoke Simulation
Download with the link above or prepare data for WDNO:
```code
cd smoke
python wave_trans_2d.py
```
training of Base-Resolution Model:
```code
bash /scripts/smoke/train_base_sim.sh
```
inference of base-resolution:
```code
bash /scripts/smoke/inf_base_sim.sh
```
training of Super-Resolution Model:
```code
bash /scripts/smoke/train_super_sim.sh
```
inference of super-resolution:
```code
bash /scripts/smoke/inf_super_sim.sh
```

## 1D Burgers' Equation Control:
Download with the link above or prepare data for WDNO:
```code
cd burgers
python wave_trans.py
```
training:
```code
bash /scripts/burgers/train_base_control.sh
```
inference:
```code
bash /scripts/burgers/eval_base_control.sh
```

## 2D Smoke Control:
Download with the link above or prepare data for WDNO:
```code
cd smoke
python wave_trans_2d.py
```
training:
```code
bash /scripts/smoke/train_base_control.sh
```
inference:
```code
bash /scripts/smoke/inf_base_control.sh
