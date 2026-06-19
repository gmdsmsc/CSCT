# Interactive Prediction Transformer without Recurrence or Convolution

## Overview

The following project extends [PredFormer](https://github.com/yyyujintang/PredFormer) to test it for interactive prediciton with a custom dataset for a jumping box (based on the video game Flappy Bird).

## Installation

```
pip install -r requirements.txt
```


## Overview

- `openstl/models:` contains model variants for comparison. Modify openstl/models/__init__.py to choose a specific model.
- `benchmarking/:` contains data generation scripts and benchmarking check scripts.


## Commands

### 1. Generate the Data

python -m tools.prepare_data.fixed_gravity  

This will create 500 new fixed_gravity data files at data/fixed_gravity.

### 2. Train

python -m tools.train --config_file configs/flappy/InteractivePredictionModel.py --dataname flappy --data_root data/fixed_gravity --res_dir work_dirs --batch_size 1 --val_batch_size 1 --epoch 30 --overwrite --lr 1e-4 --opt adamw --weight_decay 1e-2 --ex_name "flappy"

### 3. Inference

python -m tools.inference --config_file configs/flappy/InteractivePredictionModel.py --dataname flappy --data_root data/fixed_gravity --res_dir work_dirs/ --batch_size 1 --val_batch_size 1 --no_display_method_info --ex_name "flappy"

## Acknowledgments

Gratitude is owed to the following projects, by which the foundational groundwork was established and made available for further development.

- [PredFormer](https://github.com/yyyujintang/PredFormer) 
- [OpenSTL](https://github.com/chengtan9907/OpenSTL). 



