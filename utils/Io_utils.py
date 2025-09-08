import pandas as pd 
from utils.data_process import basepair_2_id_encode
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
np.random.seed(42)

def read_pairs_from_csv(file_path):
    """
    Read sgRNA/off-target pairs from a CSV file and encode them for inference/training.

    The CSV must contain the columns: {"DNA", "sg", "label"}.
    - "DNA": off-target sequence
    - "sg" : sgRNA sequence (aligned with DNA; last 3 bases are typically PAM)
    - "label": binary ground truth (0/1)
    """
    input_df = pd.read_csv(file_path)

    # 检查列是否存在
    required_cols = {"DNA", "sg", "label"}
    if not required_cols.issubset(input_df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, but got {input_df.columns.tolist()}")
    encode_list = []
    label_list = []
    sg_list = []
    for idx, row in input_df.iterrows():
        off = row['DNA'].upper()
        sg = row['sg'].upper()
        label = row['label']
        if sg[:-3] != off[:-3]:
            encode = basepair_2_id_encode(sg, off)
            encode_list.append(encode)
            label_list.append(label)
            sg_list.append(sg[:-3])
    return encode_list, label_list, sg_list

def load_pkl_dataV2(dataset_path, feature):
    """
    Load a single feature array from a '{dataset_path}'-derived pickle file.

    """    
    encode_data_file_path = dataset_path.replace(".pkl",f"_{feature}.pkl")
    if os.path.exists(encode_data_file_path):
        print(f"load encoded {dataset_name} dataset {feature}")
        data  = pkl.load(
            open(encode_data_file_path, "rb"))
    else:
        print(f"encoded {dataset_name} dataset not exist")
    return np.array(data)


def load_pkl_file(dataset_path, feature_list):
    """
    Load multiple features into a dictionary from '*_{feature}.pkl' files.

    For each feature in `feature_list`, the function loads:
      dataset_path.replace(".pkl", f"_{feature}.pkl")
    """

    data_dict = {}
    for feature in feature_list:
        data_dict[feature] = []
    for feature in feature_list:
        encode_data_file_path = dataset_path.replace(".pkl",f"_{feature}.pkl")
        data  = np.array(pkl.load(open(encode_data_file_path, "rb")))
        data_dict[feature].append(data)
    for feature in feature_list:
        data_dict[feature] = np.array(data_dict[feature]).squeeze()
        print(data_dict[feature][10])
        print(data_dict[feature][1000])
    return data_dict


def extract_data_for_train(data_dict, sgRNAS, feature_list):
    """
    Split data into train/validation subsets with label stratification.
    """    
    indices = np.arange(len(data_dict['labels']))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=data_dict['labels'])
    train_data_dict = {}

    for feature in feature_list:
        train_data_dict[feature] = data_dict[feature][train_indices]
        print(train_data_dict[feature][10])
        print(train_data_dict[feature][1000])

    val_data_dict = {}
    for feature in feature_list:
        val_data_dict[feature] = data_dict[feature][val_indices]
        print(val_data_dict[feature][10])
        print(val_data_dict[feature][1000])
    

    return train_data_dict, train_data_dict['labels'], val_data_dict, val_data_dict['labels']

def load_data_for_train(train_Data_path, val_Data_path):
    """
    High-level loader for training/inter_validation/cross_validation splits from PKL files.
    """

    feature_list = ['pos_list', 'labels', 'sg_list']

    train_data_dict = load_pkl_file(train_Data_path, feature_list)
    train_sgRNAs =  np.unique(train_data_dict['sg_list'])
    train_data, train_labels, inter_Val_data, inter_Val_labels = extract_data_for_train(train_data_dict, train_sgRNAs, feature_list)


    cross_Val_data_dict = load_pkl_file(val_Data_path, feature_list)
    cross_Val_sgRNAs =  np.unique(cross_Val_data_dict['sg_list'])
    # cross_Val_data, cross_Val_labels =extract_data_for_valV2(test_data_dict, test_sgRNAs, feature_list)
    cross_Val_data, cross_Val_labels = cross_Val_data_dict, cross_Val_data_dict['labels']
    return   {'X_train': train_data,
              'y_train': train_labels,
              'X_val': inter_Val_data,
              'y_val': inter_Val_labels,
              'X_test': cross_Val_data,
              'y_test': cross_Val_labels
              }


def output_Multiple(result):
    """
    Pretty-print per-sgRNA metrics and global averaged metrics.
    """
    print("=" * 70)
    print("Model Evaluation Results (Averaged Global Metrics)".center(70))
    print("=" * 70)

    # 1. List all SG-specific metrics first
    print(f"\n【SG-Specific Metrics】 (Total: {len(result['sg_type'])} SG sequences)")
    for i in range(len(result['sg_type'])):
        print(f"\nSG #{i+1}:")
        print(f"{'  Sequence':<25}: {result['sg_type'][i]}")
        print(f"{'  ROC-AUC':<25}: {result['sg_roc_auc'][i]:.4f}")
        print(f"{'  ROC-PRC':<25}: {result['sg_roc_prc'][i]:.4f}")

    # 2. Show global metrics with explicit averaging note
    print("\n" + "-" * 70)
    print("【Global Metrics】 (Average of all SG metrics above)")
    print(f"{'Global ROC-AUC (avg)':<25}: {result['roc_auc']:.4f}")
    print(f"{'Global ROC-PRC (avg)':<25}: {result['roc_prc']:.4f}")
    print("\n" + "=" * 70)

def output_single(sg, off, predict_list):
    """
    Pretty-print a single sgRNA/off-target prediction result.
    """   
    print(sg, off, predict_list[0])
    print("-" * 70)
    print(f"{'SG Sequence':<26} {'Off-Target Sequence':<26} {'Model Prediction':<20}")
    print("-" * 70)
    # Print values (adjust formatting based on variable type: .3f for floats, etc.)
    print(
        f"{sg:<26} "
        f"{off:<26} "  # Use .3f for 3 decimal places (adjust if off is a string)
        f"{predict_list[0]}"
    )
    print("-" * 70)