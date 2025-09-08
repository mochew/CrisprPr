from utils.data_process import sgRNA_2_numid
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.append("/wuyingfu/off_target/mymethod")  
def prepare_training(encode_list, label_list, sg_list):

    Test_data = torch.tensor(encode_list)
    Test_label = torch.tensor(label_list).long()

    Test_sg_list_encoded, Test_sg_dict = sgRNA_2_numid(sg_list)
    Test_sg_list = torch.tensor(Test_sg_list_encoded, dtype=torch.long)

    test_dataset = TensorDataset(Test_data, Test_label, Test_sg_list)
    test_loader = DataLoader(test_dataset, batch_size=50000, shuffle=False)

    return test_loader, Test_sg_dict

def sigmoid(x):
    """使用NumPy实现sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def run_inference(encode_list, label_list, sg_list):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, test_sg_dict = prepare_training(encode_list, label_list, sg_list)
    # # 1. 加载模型
    # version = 4

    # M_model = torch.load(f"/wuyingfu/CrisprPr/models/M_models/M_model{str(version)}.pth")

    # M_model = torch.jit.script(M_model)
    # M_model.save(f"/wuyingfu/CrisprPr/models/M_models/new_M_model{str(version)}.pt")
    M_model = torch.jit.load("/wuyingfu/CrisprPr/models/M_models/new_M_model0.pt") 
    M_model.to(device)
    M_model.eval()
    # M_model = torch.jit.load("/wuyingfu/CrisprPr/models/Train_models/M_models/new_M_model0.pt") 


    D_model = torch.jit.load("/wuyingfu/CrisprPr/models/D_models/new_D_model0.pt") 
    M_model.to(device)
    M_model.eval()
    # D_model = torch.jit.load("/wuyingfu/CrisprPr/models/Train_models/D_models/new_D_model0.pt") 
    predict_list = []
    label_list = []
    sg_list = []
    for batch in test_loader:
        batch_inputs, batch_labels, batch_sgRNA = batch
        batch_inputs = batch_inputs.to(torch.int8).to(device)
        batch_labels = batch_labels.to(device)
        M_outputs, _ = M_model(batch_inputs)
        D_outputs, _ = D_model(batch_inputs)

        predicted_value = (M_outputs + D_outputs) / 2
        predict_list.extend(predicted_value.detach().cpu().numpy())
        label_list.extend(batch_labels.cpu().numpy())   
        sg_list.extend(batch_sgRNA.cpu().numpy())

    return [sigmoid(sample[0]) for sample in predict_list], label_list, sg_list, test_sg_dict

