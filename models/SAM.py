# -*- coding: utf-8 -*-
# Reference the source code https://github.com/xgr19/SAM-for-Traffic-Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class Config(object):

    """配置参数"""
    def __init__(self,dataset):
        self.model_name = 'SAM'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单          -
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'      # 模型训练结果
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 0.001                
        self.require_improvement = 300000                                
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 15                                            # epoch数
        self.batch_size = 256                                           # mini-batch大小
        
        
        self.max_byte_len =  50
        self.kernel_size = [3,4]
        self.d_dim= 256
        self.dropout= 0.1
        self.filters=  256
        


class SelfAttention(nn.Module):
    """docstring for SelfAttention"""
    def __init__(self, d_dim=256, dropout=0.1):
        super(SelfAttention, self).__init__()
        # for query, key, value, output
        self.dim = d_dim
        self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = F.softmax(scores, dim=-1)
        return scores

    def forward(self, query, key, value):
        # 1) query, key, value
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention
        scores = self.attention(query, key, value)
        print(scores.shape)
        print(value.shape)
        x = torch.matmul(scores, value)
        
        # 3) apply the final linear
        x = self.linears[-1](x.contiguous())
        # sum keepdim=False
        return self.dropout(x), torch.mean(scores, dim=-2)

class OneDimCNN(nn.Module):
    """docstring for OneDimCNN"""
    # https://blog.csdn.net/sunny_xsc1994/article/details/82969867
    def __init__(self, config):
        super(OneDimCNN, self).__init__()
        self.kernel_size = config.kernel_size
        self.convs = nn.ModuleList([
                        nn.Sequential(nn.Conv1d(in_channels=config.d_dim, 
                                                out_channels=config.filters, 
                                                kernel_size=h),
                        #nn.BatchNorm1d(num_features=config.feature_size), 
                        nn.ReLU(),
                        # MaxPool1d: 
                        # stride – the stride of the window. Default value is kernel_size
                        nn.MaxPool1d(kernel_size=config.max_byte_len-h+1))
                        for h in self.kernel_size
                        ]
                        )
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        out = [conv(x.transpose(-2,-1)) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        return self.dropout(out)


class Model(nn.Module):
    """docstring for SAM"""
    # total header bytes 24
    def __init__(self, config):
        super(Model, self).__init__()
        self.posembedding = nn.Embedding(num_embeddings=50, 
                                embedding_dim=config.d_dim)
        self.byteembedding = nn.Embedding(num_embeddings=256, 
                                embedding_dim=config.d_dim)
        self.attention = SelfAttention(config.d_dim, config.dropout)
        self.cnn = OneDimCNN(config)
        self.fc = nn.Linear(in_features=config.d_dim*len(config.kernel_size),
                            out_features=config.num_classes)


    def forward(self, x, y):
        out = self.byteembedding(x) + self.posembedding(y)
        out, score = self.attention(out, out, out)
        
        out = self.cnn(out)
        out = self.fc(out)
       
        return out,score

if __name__ == '__main__':
    x = np.random.randint(0, 255, (1, 50))
    y = np.random.randint(0, 50, (1, 50))
    config = Config("/home/dl/Desktop/program/SAM/SAM/dataset/service")
    
    sam = Model(config)
    out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())


    #sam.eval()
    #out, score = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    #print(out[0], score[0])
    
    
    
