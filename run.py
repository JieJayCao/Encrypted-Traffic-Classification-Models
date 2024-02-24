
import time
import torch
import numpy as np
from train_eval import train, init_network,test

from importlib import import_module
import argparse
from memory_profiler import profile

parser = argparse.ArgumentParser(description='Encrypted Traffic Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TSCRNN, deeppacket, BiLSTM_Att, datanet')
parser.add_argument('--data', type=str, required=True, help='input dataset source')
#parser.add_argument('--test',  type=bool,default=False, required=True, help='True for Testing')
parser.add_argument('--test', type=int, default=0, help='Train or test')

args = parser.parse_args()

def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


def main():

    dataset = args.data
    model_name = args.model  
    if 'deeppacket' in model_name:
        from utils.utils_deeppacket import build_dataset, build_iterator, get_time_dif
    elif "BiLSTM" in model_name:
        from utils.utils_bilstm import build_dataset, build_iterator, get_time_dif
    elif "TSCRNN" in model_name:
        from utils.utils_tscrnn import build_dataset, build_iterator, get_time_dif
    elif "datanet" in model_name:
        from utils.utils_datanet import build_dataset, build_iterator, get_time_dif
    elif "MATEC" in model_name:
        from utils.utils_matec import build_dataset, build_iterator, get_time_dif
        
    x = import_module('models.' + model_name)

    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    
    train_iter = build_iterator(train_data, config)
    
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    
    model = x.Model(config).to(config.device)
    #init_network(model)
    
    if args.test == 1:
        print(args.test)
        print(model.parameters)
        print(get_parameter_number(model))
        train(config, model, train_iter, dev_iter, test_iter)
    else:
        print(get_parameter_number(model))
        test(config,model,test_iter)
    
if __name__ == '__main__':
    main()