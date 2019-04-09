import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.utils.extmath import softmax

import numbers
import numpy as np
import torch
#from sklearn import metrics
import torch
import torch.nn as nn

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    
    
    outputs = np.argmax(outputs, axis = 1)
    print(outputs)
    return np.sum(outputs==labels)/float(labels.size)

def AUC(outputs, labels):
    
    fpr, tpr, thresholds = metrics.roc_curve(outputs, labels)
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

#metrics_save = {'accuracy':accuracy}

