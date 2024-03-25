

# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import random

def tranHex2Dec(content):
    new = [int(i.strip("\n"),16) for i in content]
    return new


def build_dataset(config):
   
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level

    #pos = sorted(random.sample(pos,30))
    def load_dataset(path):
        contents = []
        pos = [i for i in range(200)]
        #pos = sorted(random.sample(pos,20))
        #pos =  [48, 28, 1, 8, 10, 12, 44, 19, 4, 16, 6, 49, 25, 24, 18, 22, 3, 29, 13, 15, 7, 26, 9, 45, 20, 30, 39, 32, 47, 21, 27, 42, 17, 46, 36, 2, 35, 14, 11, 34, 40, 33, 5, 41, 37, 0, 43, 31, 23, 38]
        #pos = [4,19,24,10,3,8,2,28,18,12,45,48]
        #pos = sorted(pos[:15])
        
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')

                token = tokenizer(content)
      
                token = [token[i] for i in pos] 
                contents.append((tranHex2Dec(token),pos,int(label)))
                #contents.append((tranHex2Dec(token),int(label)))
                
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        #y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
        #print(x.shape)
        #x = torch.reshape(x,(x.shape[0],50))
        
        
        return x,pos,y
        
        
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    test_list = ["ff" for i in range(50)]
    