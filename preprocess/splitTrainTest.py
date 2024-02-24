from tqdm import tqdm
import os
import  random
from utlis import ID_TO_APP,ID_TO_TRAFFIC,ID_TO_TorAPP
from utlis_ustc import PREFIX_TO_MALBEGIN_ID, ID_TO_Malben

 
  
def splitDeepPacket(inpath,outpath):
    f = open(inpath,'r')

    lines = []
    count = 0
    
    for line in open(inpath,'r'):
        count += 1
        lines.append(line)
    random.shuffle(lines)


    f_class = open(outpath+"/class.txt",'w')
    if "app" in outpath:
        for i in ID_TO_APP:
            f_class.write(ID_TO_APP[i]+"\n")
            print(i)
    elif "service" in outpath:
        for i in ID_TO_TRAFFIC:
            f_class.write(ID_TO_TRAFFIC[i]+"\n")
            print(i)
    
    

    f_train = open(outpath+"/train.txt",'w')
    f_test = open(outpath+"/test.txt",'w')
    f_dev = open(outpath+"/dev.txt",'w')

    flag1 = int((count/10)*8)
    flag2 = int((count/10)*9)

    test = []
    
    for i in range(len(lines)):
        if i <= flag1:
            f_train.write(lines[i])
        elif flag1 < i <= flag2:
            f_dev.write(lines[i])
        elif flag2 < i <= count:
            f_test.write(lines[i])
        elif i > count: 
            break



if __name__ == '__main__':
    # inpath: 预处理好的txt文件路径
    # outpath: 目标文件夹路径，即保存训练集测试集的路径


    #inpath_app = "/home/dl/Desktop/program/Deep-Packet/dataset/Datanet/app/datanet_app.txt"
    #outpath_app = "/home/dl/Desktop/program/Deep-Packet/dataset/Datanet/app/data"

    inpath_service = "../dataset/service/service.txt"
    outpath_service = "../dataset/service/data"

    splitDeepPacket(inpath_service,outpath_service)
    #splitDeepPacket(inpath_app,outpath_app)
