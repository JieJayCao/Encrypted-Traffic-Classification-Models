import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# You can refer https://github.com/munhouiani/Deep-Packet.
class Config(object):
    
    def __init__(self, dataset):
        self.model_name = 'deeppacket'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]                     
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.learning_rate = 0.001                
        self.require_improvement = 2000                         
        self.num_classes = len(self.class_list)                         
        self.num_epochs = 15                                         
        self.batch_size = 256                               
   

class Model(nn.Module):
    def __init__(self, config): 
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=200,kernel_size=4,stride=3)
        self.conv2 = nn.Conv1d(in_channels=200,out_channels=200,kernel_size=5,stride=1)
        
        self.fc1 = nn.Linear(in_features=49400, out_features=200) # ((28-5+1)/2 -5 +1)/2 = 4
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.dropout = nn.Dropout(0.05)
        self.fc3 = nn.Linear(in_features=100, out_features=50)
        self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(in_features=50, out_features= config.num_classes)
        
    def forward(self, x):        
        # hidden conv layers, conv w/ relu activation -> max pool
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.max_pool1d(out, kernel_size=2)
        out = out.reshape(-1, 200*247) 
        
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.out(out)
        return out