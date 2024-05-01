
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
import os 
import pyedflib 
#--------------------------------
Tras = {
    "Sleep stage W":0,
    "Sleep stage 1":1,
    "Sleep stage 2":2,
    "Sleep stage 3":3,
    "Sleep stage 4":4,
    "Sleep stage R":5,
    "Sleep stage ?":6,
    "Movement time":6,
}
#训练集、测试集加载数据载入------
class SleepDataset(Data.Dataset):
    def __init__(self,Path):
        self.Data = []
        self.Tag  = []
        self.FilePath = os.listdir(Path)
        self.TargetFile = {}
        for item in self.FilePath:
            FileType = 1 if item[-5] == "G" else 0
            FileCode = item.split("SC")[1].split("-")[0]
            handle   = item[2:6]
            if not handle in self.TargetFile:
                self.TargetFile[handle] = ["",""]
            if FileType:
                self.TargetFile[handle][0] = FileCode
            else:
                self.TargetFile[handle][1] = FileCode
        for Key,Value in self.TargetFile.items():
            RawData = Value[0]
            RawAno  = Value[1]
            Raw     = pyedflib.EdfReader(Path +"/SC%s-PSG.edf"      %(RawData))
            Raw     = Raw.readSignal(0)
            TAG     = pyedflib.EdfReader(Path +"/SC%s-Hypnogram.edf"%(RawAno))
            TAG     = TAG.readAnnotations()
            DataLen = (len(Raw)//(100*30))
            Point   = 0 
            for Count in range(DataLen):
                DataSplit = Raw[100*30*Count:100*30*(Count+1)]
                if (Point == len(TAG[0])-1) or (TAG[0][Point] <= Count*30 < TAG[0][Point+1]):
                    pass
                else:
                    Point += 1
                Temp_TAG = Tras[TAG[2][Point]]
                if Temp_TAG == 6:
                    DataTag   = self.Tag[-1]
                else:
                    DataTag   = Temp_TAG
                self.Data.append(np.array([DataSplit],dtype=np.float32))
                self.Tag .append(np.int64(DataTag))
        self.Data  = list(map(torch.Tensor,self.Data))
        self.Tag   = list(map(torch.tensor,self.Tag))
    def __getitem__(self,index):
        return self.Data[index],self.Tag[index]
    def __len__(self):
        return len(self.Data)
