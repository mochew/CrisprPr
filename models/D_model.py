import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
np.random.seed(42)


class Conv2DBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding='same', activation='relu', use_bias=True):
        super(Conv2DBN, self).__init__()
        if padding == 'same':
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            print(padding)
        else:
            padding = 0
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding='same', bias=use_bias)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x

class D_model(nn.Module):
    def __init__(self, initial_matrix):
        super(D_model, self).__init__()
        self.matrix = nn.Parameter(initial_matrix)
        self.branch_0 = Conv2DBN(1, 10, (1,1))
        self.branch_1 = Conv2DBN(1, 10, (2,1))
        self.branch_2 = Conv2DBN(1, 10, (3,1))
        self.branch_3 = Conv2DBN(1, 10, (5,1))
        self.blstm = nn.LSTM(41, 15, bidirectional=True, batch_first=True)
        self.blstm2 = nn.LSTM(71, 15, bidirectional=True, batch_first=True)
        self.blstm3 = nn.LSTM(30, 15, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(24*30, 80)  # 24 (sequence length) * 30 (15 * 2 for bidirectional)
        self.fc2 = nn.Linear(80, 20)
        self.dropout = nn.Dropout(0.35)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x - 1
        x = torch.flip(x, dims=[1])
        x = x.unsqueeze(dim = 2)

        rows = torch.arange(24).unsqueeze(0).repeat(x.size(0), 1).unsqueeze(2).to(x.device)
        batch_indices = torch.cat([rows, x], dim = 2)
        rows = batch_indices[..., 0]  # 形状 [B, N]
        cols = batch_indices[..., 1]  # 形状 [B, N]
        sample = self.matrix[rows, cols]
        fin_feature = sample.unsqueeze(1)
        fin_feature = fin_feature.unsqueeze(3)

        branch_0_out = self.branch_0(fin_feature)
        branch_1_out = self.branch_1(fin_feature)
        branch_2_out = self.branch_2(fin_feature)
        branch_3_out = self.branch_3(fin_feature)
        mixed = torch.cat([fin_feature.squeeze(dim=3), branch_0_out.squeeze(dim=3), branch_1_out.squeeze(dim=3), branch_2_out.squeeze(dim=3), branch_3_out.squeeze(dim=3)], dim=1)
        mixed = torch.flip(mixed.permute(0, 2, 1) , dims=[1]) 
        
        blstm_out, _ = self.blstm(mixed)
        res_embedding = torch.cat([blstm_out, mixed], dim=-1)
        blstm_out2, _ = self.blstm2(res_embedding)
        blstm_out3, _ = self.blstm3(blstm_out2)

        embedding = blstm_out3.reshape(blstm_out3.size(0), -1)
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x