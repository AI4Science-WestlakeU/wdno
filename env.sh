cd wdno

conda create -n wdno python=3.8.18 -y
conda activate wdno

pip install PyWavelets
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install --no-deps --ignore-requires-python --no-cache-dir .
cd ..
pip install --no-deps --ignore-requires-python --no-cache-dir ptwt==0.1.6

pip install -r requirements.txt
conda install numpy==1.19.0 -y
