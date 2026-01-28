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


class M_model(nn.Module):
    def __init__(self, initial_matrix):
        super(M_model, self).__init__()
        self.matrix = nn.Parameter(initial_matrix)
        self.branch_2 = Conv2DBN(1, 20, (3,1))
        self.branch_3 = Conv2DBN(1, 20, (5,1))
        self.blstm = nn.LSTM(41, 25, bidirectional=True, batch_first=True)
        self.blstm2 = nn.LSTM(50, 25, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(24*50, 80)  # 24 (sequence length) * 30 (15 * 2 for bidirectional)
        self.fc2 = nn.Linear(80, 20)
        self.dropout = nn.Dropout(0.35)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x - 1
        x = torch.flip(x, dims=[1])
        x = x.unsqueeze(dim = 2)
        cols = torch.arange(24).unsqueeze(0).repeat(x.size(0), 1).unsqueeze(2).to(x.device)
        batch_indices = torch.cat([cols, x], dim = 2)
        rows = batch_indices[..., 0]  # 形状 [B, N]
        cols = batch_indices[..., 1]  # 形状 [B, N]
        sample = self.matrix[rows, cols]
        mixed_sample = sample.unsqueeze(1)
        sample = mixed_sample.unsqueeze(3)
        branch_1_out = self.branch_1(sample)
        branch_2_out = self.branch_2(sample)
        branch_3_out = self.branch_3(sample)
        mixed = torch.cat([sample.squeeze(dim=3), branch_2_out.squeeze(dim=3), branch_3_out.squeeze(dim=3)], dim=1)
        mixed = torch.flip(mixed.permute(0, 2, 1) , dims=[1]) 

        blstm_out, _ = self.blstm(mixed)
        blstm_out2, _ = self.blstm2(blstm_out)
        embedding = blstm_out2.reshape(blstm_out2.size(0), -1)
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
