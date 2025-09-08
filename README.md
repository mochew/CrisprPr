# CrisprPr
A Hybrid-Driven Framework for CRISPR/Cas9 Off-Target Prediction with Analysis of Prior-Information Updates

## Introduction
CRISPR/Cas9 specificity is critically affected by off-target effects. However, the intricate patterns of mismatches and their combinations at off-target sites remain difficult to capture, and existing approaches show limited capacity to identify informative features. Here, we present CrisprPr, a hybrid-driven off-target prediction framework that incorporates prior knowledge to improve the characterization of off-target determinants. CrisprPr employs a synchronous updating strategy together with multi-source integration to deliver accurate and stable off-target predictions. 

## Project Structure
│── main.py                  # Main entry point (inference & analysis) <br />
│── models/                  # Model definitions (e.g., M_model, F_model) <br />
│── utils/                   # Utility functions (data preprocessing, evaluation, visualization) <br />
│── tests/                   # Testing and inference scripts <br />
│── analysis/                # Analysis module scripts (visualization, embedding updates) <br />
│── data/                    # Example datasets (sgRNA / off-target sequences)  <br />
│── train_model.py           # Training script for model development <br />
│── README.md                # Project documentation



## Create environment and install dependencies:

```
conda create -n CrisprPr python=3.8
conda activate CrisprPr
pip install -r requirements.txt
```



## 🚀 Usage

### Run inference module

There are two modes for testing:

#### 1. Input_file mode

```
 python main.py --module test --source file --input_file ./CrisprPr/data/HEK293T_test.csv 
```

#### 2. Text mode

```
 python main.py --module test --source text --sg GACTTGTTTTCATTGTTCTCAGG --off CATTTGTTTTCATTGTTCTCTGG
```

### Run analysis module

```
python main.py --module analysis --ori_path /CrisprPr/data/matrices/MTP/origin_matrix.npy  --update_path /CrisprPr/data/matrices/MTP  --output_path /CrisprPr/analysis/figs/MTP/  --seed_num 5
```

**Visualization outputs** will be saved automatically in the specified `--output_path` directory.



### Run Train models

```
python train_model.py 
```
