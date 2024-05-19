# SADA
Official PyTorch implementation of Stain-aware Domain Alignment for Imbalanced Blood Cell Classification


## Usage
### Installation
```sh
cd SADA
conda create -n SADA python=3.9
conda activate SADA
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### How to run
```sh
python train_all.py --data_dir path/to/dataset
```
