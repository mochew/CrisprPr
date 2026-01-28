# CrisprPr
A Hybrid-Driven Framework for CRISPR/Cas9 Off-Target Prediction with Analysis of Prior-Information Updates

## Introduction
CRISPR/Cas9 specificity is critically affected by off-target effects. However, the intricate patterns of mismatches and their combinations at off-target sites remain difficult to capture, and existing approaches show limited capacity to identify informative features. Here, we present CrisprPr, a hybrid-driven off-target prediction framework that incorporates prior knowledge to improve the characterization of off-target determinants. CrisprPr employs a synchronous updating strategy together with multi-source integration to deliver accurate and stable off-target predictions. The general framework of CrisprPr is as follows：
<img src="The framework of CrisprPr.png"  />  
## Project Structure
│── main.py                  # Main entry point (inference & analysis) <br />
│── models/                  # Model definitions (e.g., M_model, F_model) <br />
│── utils/                   # Utility functions (data preprocessing, evaluation, visualization) <br />
│── tests/                   # Testing and inference scripts <br />
│── analysis/                # Analysis module scripts (visualization, embedding updates) <br />
│── data/                    # Example datasets (sgRNA / off-target sequences)  <br />
│── train_model.py           # Training script for model development <br />
│── README.md                # Project documentation


## Requirements
Our tool has been tested on the following configuration on a Linux machine:<br />
python = 3.8 <br />
numpy=1.24.4 <br />
pandas=2.0.3 <br />
scikit-learn=1.3.2 <br />
scipy=1.10.1 <br />
seaborn=0.13.2 <br />
torch=1.13.1 <br />
torchvision=0.14.1 <br />

## Download
(1) Git Clone
```
git clone https://github.com/mochew/CrisprPr.git
```
(2) Download the ZIP file from https://github.com/mochew/CrisprPr, then extract it to your desired location.
```
unzip CrisprPr-main.zip
```

## Create environment and install dependencies:

```
conda create -n CrisprPr python=3.8
conda activate CrisprPr
pip install -r requirements.txt
```
## Dataset Overview
- **Training set (CHANGEseq)**  
  `./data/train_data/`

- **Validation set (TTISS-CT)**  
  `./data/val_data/`

- **Test sets**  
  `./data/test_data/`


## 🚀 Usage

### Run inference module

There are two modes for testing:

#### 1. Input_file mode
--initialization selects the model variant based on how missing values in the prior knowledge were initialized before training: <br />
   binary: model trained with prior knowledge where missing values were filled with 0/1; <br />
   mean-value: model trained with prior knowledge where missing values were filled with the global mean. <br />
```
 python main.py --module test --source file  --initialization binary --input_file ./data/test_sample.csv 
```

#### 2. Text mode
--initialization selects the model variant based on how missing values in the prior knowledge were initialized before training: <br />
   binary: model trained with prior knowledge where missing values were filled with 0/1; <br />
   mean-value: model trained with prior knowledge where missing values were filled with the global mean. <br />

```
 python main.py --module test --source single --initialization binary --sg GACTTGTTTTCATTGTTCTCAGG --off CATTTGTTTTCATTGTTCTCTGG
```

### Run analysis module

```
python main.py --module analysis --ori_path ./data/matrices/MTP/origin_matrix.npy  --update_path ./data/matrices/MTP/  --output_path ./analysis/figs/MTP/  --seed_num 5
```

**Visualization outputs** will be saved automatically in the specified `--output_path` directory.



### Run Train models

```
python train_model.py 
```
