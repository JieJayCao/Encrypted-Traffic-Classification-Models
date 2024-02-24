
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

    """配置参数"""
    def __init__(self,dataset):
        self.model_name = 'MATEC'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                   
        self.test_path = dataset + '/data/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              
        self.vocab_path = dataset + '/data/vocab.pkl'                               
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'       
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.dropout = 0.1                                             
        self.require_improvement = 10000                               
        self.num_classes = len(self.class_list)                        
        self.n_vocab = 0                                               
        self.num_epochs = 50                                         
        self.batch_size = 128                                          
        self.pad_size = 786                                              
        self.learning_rate = 0.0005                                      
        self.embed =  432          
        self.dim_model = 432
        self.num_head = 3
        self.num_encoder = 2


'''Attention Is All You Need'''
class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
       

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        return out

class Add_Norm(nn.Module):
    def __init__(self, dim_model):
        super(Add_Norm, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, origin_x, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = out + origin_x  
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out 
    

    
class Feed_Forward(nn.Module):
    def __init__(self, dim_model):
        super(Feed_Forward, self).__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Feed_Forward(dim_model)
        self.add_norm1 = Add_Norm(dim_model)
        self.add_norm2 = Add_Norm(dim_model)
        
    def forward(self, x):
        
        out = self.attention(x)
        out = self.add_norm1(x,out)
        out_ = self.feed_forward(out)
        out = self.add_norm2(out,out_)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(config.pad_size,config.dim_model)
        self.encoder1 = Encoder(config.dim_model, config.num_head, config.dropout)
        self.encoder2 = Encoder(config.dim_model, config.num_head, config.dropout)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        self.fc2 = nn.Linear(1296, config.num_classes)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = torch.flatten(out,start_dim=1)
        out = self.dropout(out)
        out = self.fc2(out)
    
        return out
    
if __name__ == '__main__':
    x1 = torch.rand(10,1,786)
    x2 = torch.rand(10,1,786)
    x3 = torch.rand(10,1,786)
    x = (x1,x2,x3)
    config = Config("/home/dl/Desktop/program/Deep-Packet/dataset/Datanet/service")
    matec = Model(config)
    y= matec(x)
    print(y)