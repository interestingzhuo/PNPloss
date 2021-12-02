# PNP Loss in PyTorch

---
## What can I find here?

This repository contains all code and implementations used in:

```
Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough
```
accepted to AAAI 2022

### Requirements:

* PyTorch 1.2.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels
(9) Run the scripts!
```

### Datasets:
Data for
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)


* For Stanford Online Products:
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

Assuming your folder is placed in e.g. `<$datapath/sop>`, pass `$datapath` as input to `--source`.

### Training:
Training is done by using `main.py` and setting the respective flags, all of which are listed and explained in `parameters.py`.

**A basic sample run using the best parameters would like this**:

```
python main.py --loss PNP  --seed 0 --bs 384 --data_sampler class_random --samples_per_class 4 --arch resnet50_frozen_normalize --source ../retrieval_dataset --n_epochs 400 --lr 1e-5 --embed_dim 512 --evaluate_on_gpu --dataset online_products --variant PNP-D_q --alpha 4
```
## Paper
If you find this work useful, please consider citing:
```
@InProceedings{Zhuo2022,
  author       = "Zhuo Li and Weiqing Min and Jiajun Song and Yaohui Zhu and Liping Kang and Xiaoming Wei and Xiaolin Wei and Shuqiang Jiang",
  title        = "Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough",
  booktitle    = "AAAI Conference on Artificial Intelligence (AAAI 2022)",
  year         = "2022",
}
```
