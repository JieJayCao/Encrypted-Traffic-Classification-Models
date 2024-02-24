# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
import wandb
from memory_profiler import profile

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


        
def train(config, model, train_iter, dev_iter, test_iter):

    wandb.init(project=config.model_name+"-"+config.train_path.split("/")[-3])
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.num_epochs,
    "batch_size": config.batch_size
    }

    start_time = time.perf_counter()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #lambda1 = lambda epoch:np.sin(epoch)/epoch
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    start_time = time.perf_counter()
    for epoch in range(config.num_epochs): 
        print('Epoch [{}/{}]'.format(epoch + 1,config.num_epochs))
        
        for i,(traffic, labels) in enumerate(train_iter): 
            
            preds,_ = model(traffic)
            loss = F.cross_entropy(preds, labels)
            #loss = Loss(preds,labels)
           
            optimizer.zero_grad()               
            loss.backward()       
            optimizer.step()
            #scheduler.step()       

          
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(preds.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                wandb.log({"train_loss":  loss.item()})
                wandb.log({"train_acc":  train_acc})
                #wandb.log({"dev_loss":  dev_loss})
                #wandb.log({"dev_acc":  dev_acc})
                model.train()
                wandb.watch(model)
                
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc,improve))
             
            total_batch += 1
            if total_batch - last_improve > 200000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        
        if flag:
            break
    end_time = time.perf_counter()
    print(end_time-start_time)
    test(config, model, test_iter)



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    #print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    start_time = time.time()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
   
    with torch.no_grad():
        for traffic,labels in data_iter:
            
            outputs,_ = model(traffic)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()

            predict_ = torch.softmax(outputs,dim=1)
            predict_ = predict_.cpu().numpy()

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    time_dif = get_time_dif(start_time)
    if test == True:
        print("####", time_dif)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
