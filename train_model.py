from models.M_model import M_model
from models.D_model import D_model
from utils.train_utils import EarlyStopping_prc, remove_on_target
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils.Io_utils import load_data_for_train
import time
from utils.data_process import sgRNA_2_numid
from utils.evaluation import compute_AUPRC_and_AUROC_scores
from collections import Counter, defaultdict
model_config = {
    'Lr' :0.0001,
    'epoch' :300,
    'batch_size' : 50000,
    'model_seed': 168,
    'patience': 50,
    'epoches': 300
}

def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch CPU 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    torch.backends.cudnn.deterministic = True  # 确保 CUDNN 的确定性行为
    torch.backends.cudnn.benchmark = False  # 禁止 CUDNN 自动优化


def convert_to_tensor(train_list, inter_Val_list, cross_Val_list):
        train_data, train_labels = train_list
        inter_Val_data, inter_Val_labels, inter_Val_sg_list = inter_Val_list
        cross_Val_data, cross_Val_labels, cross_Val_sg_list = cross_Val_list
    
        train_data = np.concatenate([train_data, inter_Val_data], axis=0)
        train_labels = np.concatenate([train_labels, inter_Val_labels], axis=0)
        print(f"combined_train_data: {train_data.shape}, combined_train_labels:{train_labels.shape}")
        train_data = torch.tensor(train_data)
        train_y = torch.tensor(train_labels).long()
        train_dataset = TensorDataset(train_data, train_y)
        train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)

        cross_Val_sg_list_encoded, cross_Val_sg_dict = sgRNA_2_numid(cross_Val_sg_list)
        cross_Val_sg_list = torch.tensor(cross_Val_sg_list_encoded, dtype=torch.long)
        cross_Val_data = torch.tensor(cross_Val_data)
        cross_Val_y = torch.tensor(cross_Val_labels).long()
        cross_Val_dataset = TensorDataset(cross_Val_data, cross_Val_y, cross_Val_sg_list)
        cross_Val_loader = DataLoader(cross_Val_dataset, batch_size=model_config['batch_size'], shuffle=False)

        return train_loader, cross_Val_loader, cross_Val_sg_dict



def load_data(train_Data_path, val_Data_path):
    data = load_data_for_train(train_Data_path, val_Data_path)
    X_train, train_labels = data['X_train'], data['y_train']
    X_Inter_val, inter_Val_labels = data['X_val'], data['y_val']
    X_Cross_val, cross_Val_labels = data['X_test'], data['y_test']
    
    train_Pos_list =  X_train['pos_list']
    inter_Val_Sg_list, inter_Val_Pos_list =  X_Inter_val['sg_list'], X_Inter_val['pos_list']
    cross_Val_Sg_list, cross_Val_Pos_list = X_Cross_val['sg_list'], X_Cross_val['pos_list']

    train_data, inter_Val_data, cross_Val_data  = remove_on_target([train_Pos_list, train_labels], [inter_Val_Pos_list, inter_Val_labels, inter_Val_Sg_list],\
                     [cross_Val_Pos_list, cross_Val_labels, cross_Val_Sg_list])

    return train_data, inter_Val_data, cross_Val_data 

def train_model(Train_data_path, val_data_path, train_model_path, model_type, prior_matrix_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    train_data, inter_Val_data, intra_Val_data = load_data(Train_data_path, val_data_path)
    set_seed(model_config['model_seed'])
    if model_type == 'M':
        M1_matrix = torch.tensor(np.load(prior_matrix_path), dtype=torch.float32)
        model = M_model(M1_matrix)
    elif model_type == 'D':
        D_matrix =  torch.tensor(np.load(prior_matrix_path), dtype=torch.float32)
        model = D_model(D_matrix)
    print(model_config["model_seed"])
    print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_config['Lr']) 
    train_loader, cross_Val_loader, cross_Val_sg_dict = convert_to_tensor(train_data, inter_Val_data, intra_Val_data)
    early_stopping = EarlyStopping_prc(patience=model_config['patience'], verbose=True, path=train_model_path)
    epoches = model_config['epoches']
    for epoch in range(epoches):
        total_loss = 0
        model.train()
        time1 = time.time()
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(torch.int8).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_average_loss = total_loss / len(train_loader)

        model.eval()
        # 外部验证集
        total_cross_val_loss = 0
        cross_val_predicted_values = []
        cross_val_labels = []
        cross_val_sg_list = []
        for val_batch in cross_Val_loader:
            inputs, labels, sg_list = val_batch
            inputs = inputs.to(torch.int8).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            predit_value = outputs.clone()
            cross_val_loss = criterion(outputs.squeeze(), labels.float())
            total_cross_val_loss += cross_val_loss.item()
            
            cross_val_predicted_values.extend(predit_value.detach().cpu().numpy())
            cross_val_labels.extend(labels.detach().cpu().numpy())   
            cross_val_sg_list.extend(sg_list.detach().cpu().numpy()) 
        cross_val_results = compute_AUPRC_and_AUROC_scores(cross_val_labels, cross_val_predicted_values, cross_val_sg_list, cross_Val_sg_dict)
        average_cross_val_loss = total_cross_val_loss / len(cross_Val_loader)

        time2 = time.time()
        print(f'Epoch [{epoch+1}/{epoches}], Average Training Loss: {train_average_loss:.4f}, time: {time2-time1} ')
        early_stopping(cross_val_results['roc_prc'], model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def main():
    
    train_Data_path = "./data/train_data/CHANGEseq_encoded.pkl"
    val_Data_path = "./data/val_data/ttiss_encoded.pkl"
    # train_Model_path = f"./models/M_models/new_M_model{model_config['model_seed']}.pth"
    # model_type = "M"
    # prior_matrix_path = "./models/M_models/M1_matrix_dic_D9_new.npy"

    train_Model_path = f"./models/D_models/new_D_model{model_config['model_seed']}.pth"  
    model_type = "D"
    prior_matrix_path = "./models/D_models/crisot_score_param_new.npy"    
    train_model(train_Data_path, val_Data_path, train_Model_path, model_type, prior_matrix_path)


if __name__ == "__main__":
    main()



