import torch
import torch.nn as nn
import torch.functional as F 
import numpy as np

Fs = 512 # sampling frequency

device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ECA1DModule(nn.Module):
    def __init__(self, k_size=3):
        super(ECA1DModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        # Multi-scale information fusion
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)

class CNN1(nn.Module): # smaller filter sizes to learn temporal information
    def __init__(self,training=False):
        super(CNN1,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=Fs//2,stride=Fs//16,bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=8,stride=8)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=64 ,kernel_size=8,out_channels=128,stride=1,bias=False)
        self.conv3 = nn.Conv1d(in_channels=128,kernel_size=8,out_channels=128,stride=1,bias=False)
        self.conv4 = nn.Conv1d(in_channels=128,kernel_size=8,out_channels=128,stride=1,bias=False)
        self.pool2 = nn.MaxPool1d(kernel_size=4,stride=4)
        self.bn1   = nn.BatchNorm1d(num_features=128)
        self.bn2   = nn.BatchNorm1d(num_features=128)
        self.bn3   = nn.BatchNorm1d(num_features=128)
        self.bn4   = nn.BatchNorm1d(num_features=128)
        self.RL1   = nn.ReLU(inplace=True)
        self.RL2   = nn.ReLU(inplace=True)
        self.RL3   = nn.ReLU(inplace=True)
        self.RL4   = nn.ReLU(inplace=True)
        self.AtLy1 = ECA1DModule()
        self.AtLy2 = ECA1DModule()

    def forward(self,x): 
        x = self.RL1(self.bn1(self.conv1(x)))
        x = self.dropout(self.pool1(x))
        x = self.RL2(self.bn2(self.conv2(x)))
        #x = self.AtLy1(x)
        x = self.RL3(self.bn3(self.conv3(x)))
        x = self.RL4(self.bn4(self.conv4(x)))
        #self.AtLy2(x)
        #x = self.pool2(x)
        return x 
        
class CNN2(nn.Module): # larger filter sizes to learn frequency information
    def __init__(self,training=False):
        super(CNN2,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=int(Fs*4),stride=Fs//2,bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=4,stride=4)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=64 ,kernel_size=6,out_channels=128,stride=1,bias=False)
        self.conv3 = nn.Conv1d(in_channels=128,kernel_size=6,out_channels=128,stride=1,bias=False)
        self.conv4 = nn.Conv1d(in_channels=128,kernel_size=6,out_channels=128,stride=1,bias=False)
        self.pool2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.RL1 = nn.ReLU()
        self.RL2 = nn.ReLU()
        self.RL3 = nn.ReLU()
        self.RL4 = nn.ReLU()
        self.AtLy1 = ECA1DModule()
        self.AtLy2 = ECA1DModule()

    def forward(self,x):  
        x = self.RL1(self.bn1(self.conv1(x)))
        x = self.dropout(self.pool1(x))
        x = self.RL2(self.bn2(self.conv2(x)))
        #x = self.AtLy1(x)
        x = self.RL3(self.bn3(self.conv3(x)))
        x = self.RL4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        #x = self.AtLy2(x)
        return x 

class DeepSleepNet(nn.Module):
    def __init__(self,training=False):
        super(DeepSleepNet,self).__init__()
        self.training    = training
        self.cnn1        = CNN1()
        self.cnn2        = CNN2()
        self.dropout     = nn.Dropout(0.8)
        #self.hidden_layer_size = 256
        #self.lstm        = nn.LSTM(256,256,2,bidirectional=True)
        self.fc          = nn.Linear(256,128)
        self.final_layer = nn.Linear(128,6)
        self.Sig         = nn.Sigmoid()
        #self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                    torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self,x):
        #----------------------------------
        print(np.shape(x))
        temp1 = x.clone()
        x     = self.cnn1(x)
        temp1 = self.cnn2(temp1)
        x     = torch.cat((x,temp1),dim=0)
        x     = self.fc(x)
        #x     = x.view(-1,self.num_flat_features(x))
        x     = self.dropout(x)
        x     = self.final_layer(x)
        x     = self.Sig(x)
        #temp2 = x.clone()
        #temp2 = self.fc(temp2)
        #----------------------------------
        #x     = self.dropout(self.lstm(x))
        #x     = torch.add(x,temp2)
        #x     = self.dropout(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

