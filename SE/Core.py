import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import cv2
from   copy import deepcopy
import os
from   NeuralNetworkModel import *
from   DataSet            import SleepDataset as SleepDataset
#------------------------------------------
BATCH_SIZE = 1500
LR         = 0.00001
EPOCHS     = 240  
device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#------------------------------------------

#-------------------------------------
TrainSet      = SleepDataset("/root/TRAIN")
print("Train set size",len(TrainSet))
TestSet       = SleepDataset("/root/TEST")
print("Test set size",len(TestSet))
#-------------------------------------
Train_loader = torch.utils.data.DataLoader(dataset=TrainSet,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
Test_loader  = torch.utils.data.DataLoader(dataset=TestSet ,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
#-------------------------------------
#Model     = MobileNetV3(n_class=6, input_size=128, dropout=0.8, mode='large', width_mult=1.0).to(device)
Model     = Sleep_Net().to(device)
#Model    = torch.load("/root/Sleep_Net_2.pt").to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
#-------------------------------------
# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#-------------------------------------
def tarin():
    Lost_tarin = []
    Lost_test  = []
    total_step = len(Train_loader)
    #curr_lr = LR
    for epoch in range(EPOCHS):
        LOSS_SUM = 0
        for items, labels in Train_loader:
            # 将数据放入GPU
            items   = items.to(device)
            labels  = labels.to(device).squeeze(-1)
            outputs = Model(items)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS_SUM += (loss.item()/total_step)
        print(">>>训练集|批次[{}/{}]|损失值:{:.4f}>".format(epoch + 1, EPOCHS,LOSS_SUM))
        Lost_tarin.append(LOSS_SUM)
        #----------------------------
        if (epoch) % 10 == 0:
            global LR
            LR/=1.5
            update_lr(optimizer,LR)
        #-----------------------------
        try:
            with open("/root/Record_Train.txt","a+") as File:
                File.write("%.3f\n"%(LOSS_SUM ))
        except:
            pass
        if (epoch) % 2 == 0: 
            Test_Result = test()
            Lost_test.append(Test_Result)
            with open("/root/Record_Test.txt","a+") as File:
                File.write("%.5f\n"%(Test_Result))
            torch.save(Model,"/root/Sleep_Net_2.pt")
        #-----------------------------
    torch.save(Model,"/root/Sleep_Net_2.pt")
    os.system("/root/shutdown.sh")
#--------------------------------------------
def test():
    Model.eval()
    with torch.no_grad():
        total     = 0
        RightTime = 0 
        for items, labels in Test_loader:
            items   = items .to(device)
            labels  = labels.to(device)
            outputs = Model(items)
            output  = outputs.cpu().tolist()
            label   = labels .cpu().tolist()
            #print("|",np.argmax(output[0]),label[0],"|")
            for i,item in enumerate(output):
                
                if np.argmax(item) == label[i]:
                    RightTime += 1
                total += 1
    print(">>>测试集|正确率：%.5lf"%(RightTime/total))
    return (RightTime/total)

if __name__ == "__main__":
    tarin()
        