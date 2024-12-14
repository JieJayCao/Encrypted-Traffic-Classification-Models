import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

 
    def __init__(self, dataset):
        self.model_name = 'TSCRNN'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]                        
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.learning_rate = 0.001                                       
        self.num_classes = len(self.class_list)                  
        self.num_epochs = 30                                            
        self.batch_size = 128                                           
   

class Model(nn.Module):
    def __init__(self, config): 
    
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 15, out_channels= 64 , kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64, affine = True)
        self.conv2 = nn.Conv1d(in_channels= 64, out_channels= 64 , kernel_size=3, stride=1,  padding=1)
        self.bn2 = nn.BatchNorm1d(64, affine = True)
        self.lstm = nn.LSTM(375, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=512, out_features=config.num_classes) 
      
        
    def forward(self, x):        
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)

        out,_ = self.lstm(out)
        out = self.dropout(out)
        
        out = self.out(out[:, -1, :]) 
        return out
