"""Defines the neural network, and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import softmax

import numbers
import numpy as np
import torch
from sklearn import metrics
import torch
import torch.nn as nn

class SimpleCNN(torch.nn.Module):    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 3)
        self.norm1 = nn.BatchNorm3d(64,eps=0.001, momentum=0.1, affine=True)
        
        self.conv2 = nn.Conv3d(64, 128, 2)
        self.norm2 = nn.BatchNorm3d(128,eps=0.001, momentum=0.1, affine=True)
        
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.norm3 = nn.BatchNorm3d(256,eps=0.001, momentum=0.1, affine=True)
        
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.norm4 = nn.BatchNorm2d(256,eps=0.001, momentum=0.1, affine=True)
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.conv7 = nn.Conv2d(512, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.norm5 = nn.BatchNorm2d(512,eps=0.001, momentum=0.1, affine=True)

        
        self.linear1 = nn.Linear(4096, 4096)
        self.linear2 = nn.Linear(4096, 1)
        self.norm1d = nn.BatchNorm1d(4096,eps=0.001, momentum=0.1, affine=True)


        
        self.maxpool1 = nn.MaxPool2d((2,2))
        self.maxpool2 = nn.MaxPool3d((2,2,2))
        
        
    def forward(self, x):
        print(x.size())
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.maxpool1(x)
        x = F.relu(self.norm4(self.conv4(x)))
        x = self.maxpool1(x)
        print(x.shape)
        x = F.relu(self.norm5(self.conv5(x)))
        x = self.maxpool1(x)
        x = F.relu(self.norm5(self.conv6(x)))
        x = self.maxpool1(x)
        print(x.shape)
        x = F.relu(self.norm5(self.conv7(x)))
        x = self.maxpool1(x)
        x = F.relu(self.norm5(self.conv8(x)))
        x = self.maxpool1(x)
        print(x.size())
        x = self.linear1(x)
        return(x)

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs)
    return np.sum(outputs==labels)/float(labels.size)

def AUC(outputs, labels):

    outputs = np.round(softmax(np.array([outputs])))
    labels = np.array([labels])
    
    outputs = list(outputs[0])
    labels = list(labels[0])
    
    print(outputs)
    print(labels)
    
    fpr = [ 0.5, 0.5]
    tpr = [ 0.5, 0.5]

    fpr, tpr, thresholds = metrics.roc_curve(outputs, labels)
    print(fpr)
    print(tpr)
    tpr[np.isnan(tpr)] = 0.5
    fpr[np.isnan(fpr)] = 0.5 
    if(len(tpr) <= 1):
        tpr=[0.5, 0.5]
    if(len(fpr) <= 1):
        fpr=[0.5,0.5]
    print(metrics.auc(fpr,tpr))
    return metrics.auc(fpr,tpr)

def fpr(outputs, labels):
    outputs = np.round(softmax(np.array([outputs])))
    labels = np.array([labels])
    
    outputs = list(outputs[0])
    labels = list(labels[0])
    fpr, tpr, thresholds = metrics.roc_curve(outputs, labels)
    fpr = np.nanmean(fpr)
    return fpr

def tpr(outputs, labels):
    outputs = np.round(softmax(np.array([outputs])))
    labels = np.array([labels])
    
    outputs = list(outputs[0])
    labels = list(labels[0])
    fpr, tpr, thresholds = metrics.roc_curve(outputs, labels)
    tpr = np.nanmean(tpr)
    return tpr

  
#maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics_save= {
    'accuracy': accuracy,
    'AUC': AUC,
    'fpr': fpr,
    'tpr':tpr
}

