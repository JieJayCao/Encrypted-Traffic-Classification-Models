import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):


    def __init__(self, dataset):
        self.model_name = 'datanet'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]                        
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.learning_rate = 0.01                                           
        self.num_classes = len(self.class_list)                         
        self.num_epochs = 100                                           
        self.batch_size = 128                                        
   

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
    
        self.fc1 = nn.Linear(in_features=1480, out_features=128) 
        #self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        #self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(in_features=32, out_features= config.num_classes)
        
    def forward(self, x):        
    
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        return 