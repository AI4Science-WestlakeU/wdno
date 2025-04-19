# WDNO: Wavelet Diffusion Neural Operator (ICLR 2025)

[arXiv](https://arxiv.org/abs/2412.04833) ｜ [Paper](https://openreview.net/forum?id=FQhDIGuaJ4)

Official repo for the paper [Wavelet Diffusion Neural Operator](https://arxiv.org/abs/2412.04833).<br />
[Peiyan Hu*](https://peiyannn.github.io/), [Rui Wang*](https://scholar.google.ca/citations?hl=zh-CN&user=8VTaeFwAAAAJ), [Xiang Zheng](https://openreview.net/profile?id=~Xiang_Zheng5), [Tao Zhang](https://zhangtao167.github.io), [Haodong Feng](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en), [Ruiqi Feng](https://weenming.github.io/), [Long Wei](https://longweizju.github.io/), [Yue Wang](https://www.microsoft.com/en-us/research/people/yuwang5/), [Zhi-Ming Ma](http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html), [Tailin Wu†](https://tailin.org/).<br />
ICLR 2025. 

We introduce Wavelet Diffusion Neural Operator (WDNO), a novel method for generative PDE simulation and control, to address diffusion models' challenges of modeling system states with abrupt changes and generalizing to higher resolutions.

Framework of WDNO:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/wdno/blob/main/fig/figures1.png" align="center" width="900" ></a>


## Environment
Run the following commands to install dependencies. In particular, when running the 2D control task, the Python version must be 3.8 due to the requirement of the Phiflow software.

```code
bash env.sh
```

## Datasets
The datasets can be downloaded respectively in [link](https://drive.google.com/drive/folders/1W1tbQ7ltIDEQdHzUarFo9EYMe4ngxXcz).
Please place the corresponding datasets in the `data` folder under the experiment directory.

## Checkpoints
The checkpoints can be downloaded respectively in [link](https://drive.google.com/drive/u/2/folders/1qjYXG53Y6cSK24EeyKLqab3kAzs961we).
Please place the corresponding checkpoints in the `results` folder under the experiment directory.

## 1D Burgers' Equation Simulation
Prepare data for WDNO:
```code
cd burgers
python wave_trans.py
```
Training of Base-Resolution Model:
```code
bash /scripts/burgers/train_base_sim.sh
```
Inference of Base-Resolution Model:
```code
bash /scripts/burgers/eval_base_sim.sh
```
Training of Super-Resolution Model:
```code
bash /scripts/burgers/train_super_sim.sh
```
Inference of Super-Resolution Model:
```code
bash /scripts/burgers/eval_super_sim.sh
```

## 2D Smoke Simulation
Prepare data for WDNO:
```code
cd smoke
python wave_trans_2d.py
```
Training of Base-Resolution Model:
```code
bash /scripts/smoke/train_base_sim.sh
```
Inference of base-resolution:
```code
bash /scripts/smoke/inf_base_sim.sh
```
Training of Super-Resolution Model:
```code
bash /scripts/smoke/train_super_sim.sh
```
Inference of super-resolution:
```code
bash /scripts/smoke/inf_super_sim.sh
```

## 1D Burgers' Equation Control:
Prepare data for WDNO:
```code
cd burgers
python wave_trans.py
```
Training:
```code
bash /scripts/burgers/train_base_control.sh
```
Inference:
```code
bash /scripts/burgers/eval_base_control.sh
```

## 2D Smoke Control:
Prepare data for WDNO:
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
```

## Related Projects
* [CL-DiffPhyCon](https://github.com/AI4Science-WestlakeU/CL_DiffPhyCon) (ICLR 2025): We introduce an improved, closed-loop version of DiffPhyCon. It has an asynchronous denoising schedule for physical systems control tasks and achieves closed-loop control with significant speedup of sampling efficiency.

* [WDNO](https://github.com/AI4Science-WestlakeU/wdno) (ICLR 2025): We propose Wavelet Diffusion Neural Operator (WDNO), a novel method for generative PDE simulation and control, to address diffusion models' challenges of modeling system states with abrupt changes and generalizing to higher resolutions, via performing diffusion in the wavelet space.

* [CinDM](https://github.com/AI4Science-WestlakeU/cindm) (ICLR 2024 spotlight): We introduce a method that uses compositional generative models to design boundaries and initial states significantly more complex than the ones seen in training for physical simulations.

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
hu2025wavelet,
title={Wavelet Diffusion Neural Operator},
author={Peiyan Hu and Rui Wang and Xiang Zheng and Tao Zhang and Haodong Feng and Ruiqi Feng and Long Wei and Yue Wang and Zhi-Ming Ma and Tailin Wu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=FQhDIGuaJ4}
}
```
