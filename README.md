# 2023-Stanford-Ribonanza-RNA-Folding
This repository contains my code for <b>Ayaan's part in the 5th place solution</b> of [the Stanford Ribonanza RNA Folding competition 2023](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/overview). All training was conducted on two GeForce RTX 4090 GPUs. The 5-fold ensemble alone achieved a leaderboard score of 0.13963 public and 0.14258 private.

You can check out our team's solution [here](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460250).

# File Description
```
├── data
│   ├── train_data_processed_ALL.parquet (train dataset)
│   ├── test_sequences_processed_ALL.parquet (test dataset)
│   └── submission_for_pseudo_v4_095.parquet (pseudo dataset)
├── exp
│   └── trainer.py
├── main
│   ├── bottle.py
│   ├── data.py
│   ├── modules.py
│   └── utils.py
├── eda.ipynb
└── infer-list.ipynb
```


# Setup
First, clone this repository:
```
git clone https://github.com/ehdgnsdl/2023-Stanford-Ribonanza-RNA-Folding.git
cd 2023-Stanford-Ribonanza-RNA-Folding
```
The code is tested for Python 3.11.5 and the packages listed in environment.yml. The basic requirements are PyTorch and Torchvision. The easiest way to get going is to install the dp_gan conda environment via
```
conda env create --file environment.yml
conda activate stanford_rna
```

# Data Preparation
The competition dataset can be downloaded [here](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data).

1. You can utilize the `eda.ipynb` file. (Refine the `training dataset` for use in the model.) <br>

2. You save the Refined datasets in the `/data` folder. (train, test, pseudo dataset)


# Training
```
cd exp
python trainer.py
```

# Inference
You can utilize the `infer-list.ipynb` file.


# Reference
I developed this based on [@sroger's code](https://github.com/s-rog/StanfordRibonanza2023).
