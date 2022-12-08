import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

"""
Identification of Encrypted Traffic Through Attention Mechanism Based Long Short Term Memory [IEEE Transactions on Big Data'19]
"""

class Config(object):

    # Configs 
    def __init__(self, dataset):
        self.model_name = 'BiLSTM_Att'
        self.train_path = dataset + '/data/train.txt'                                # Training set path
        self.dev_path = dataset + '/data/dev.txt'                                    # Dev set path
        self.test_path = dataset + '/data/test.txt'                                  # Testing set path
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # Classes file path         
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'      # Model saved path 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device 
        self.learning_rate = 0.001                
        self.num_classes = len(self.class_list)                        
        self.num_epochs = 30                                            
        self.batch_size = 64                                          
        self.dropout = 0.1
        self.hidden_size = 100

class Model(nn.Module):
    def __init__(self, config): # default grayscale
        super(Model, self).__init__()
        self.lstm = nn.LSTM(1500, config.hidden_size, 2, bidirectional=True, batch_first=True)
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1  = nn.Linear(config.hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

      
    def forward(self, x):
        H, _ = self.lstm(x)  
        alpha = F.softmax(torch.matmul(H, self.w), dim=1).unsqueeze(-1)  
        out = H * alpha  
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)  
        return out
