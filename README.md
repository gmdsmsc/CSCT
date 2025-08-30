# Interactive Prediction Transformer without Recurrence or Convolution



## Installation

```
# Exp Setting: PyTorch: 2.1.0+ Python 3.10
conda env create -f environment.yml  
conda activate predformer
pip install -e .
pip install tensorboard einops
```

## Overview

- `openstl/models:` contains model variants for comparison. Modify openstl/models/__init__.py to choose a specific model.
- `benchmarking/:` contains data generation scripts and benchmarking check scripts.


## Commands
### Train

python tools/train.py --config_file configs/flappy/PredFormer.py --dataname flappy --data_root data --res_dir work_dirs --batch_size 1 --val_batch_size 16 --epoch 30 --overwrite --lr 1e-3 --opt adamw --weight_decay 1e-2 --ex_name "flappy"  --tb_dir logs_tb/03_08

### Inference

python tools/inference.py --config_file configs/flappy/PredFormer.py --dataname flappy --data_root data --res_dir work_dirs/ --batch_size 1 --val_batch_size 1 --no_display_method_info --ex_name "flappy"

## Acknowledgments

Gratitude is owed to the following projects, by which the foundational groundwork was established and made available for further development.

- [PredFormer](https://github.com/yyyujintang/PredFormer) 
- [OpenSTL](https://github.com/chengtan9907/OpenSTL). 



