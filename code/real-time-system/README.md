# CAUSAL-GAT

This repo extends the Code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf) by including a causal-based connection learning approach

# Installation
### Requirements
* Python >= 3.6

* cuda == 11.1
* [Pytorch==1.10.1](https://pytorch.org/)
* [PyG: torch-geometric==2.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

* cuda == 11.8/12.1
* [Pytorch==2.0.0](https://pytorch.org/)
* [PyG: torch-geometric==2.0.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* Find install_latest.sh

### Install packages
```
    # run after installing correct Pytorch package
    bash install.sh
```

### Quick Start
Run to check if the environment is ready
```
    bash run.sh cpu msl
    # or with gpu
    bash run.sh <gpu_id> msl    # e.g. bash run.sh 1 msl
```


# Usage
We use part of msl dataset(refer to [telemanom](https://github.com/khundman/telemanom)) as demo example. 

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/msl/

data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)

## Run
```
    # using gpu
    bash run.sh <gpu_id> <dataset>

    # or using cpu
    bash run.sh cpu <dataset>
```
You can change running parameters in the run.sh.

# Others
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/)

./pretrained/{dir_path}/best_{datestr}.pt
python TestingAndVisualization.py -dataset swat -save_path_pattern swat -slide_stride 1 -slide_win 5 -batch 1 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -significance_level 0.05 -date_str 09_14-09-26-02 -device cpu

