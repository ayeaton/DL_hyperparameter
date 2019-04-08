"""Train the model"""

#################################################################################
#   Utils from CS230 Deep Learning at Stanford
#     TAs: Surag Nair, Guilluame Genthial, Olivier Moindrot
#     https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
################################################################################

import argparse
import logging
import os
import math
import glob

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import features.utils as utils

from features.val import evaluate
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision
import features.metrics_code as metrics_code

USE_CUDA = 1

def train(model, train_loader, metrics_save, loss_func, optimizer, inception):
    """
    Train the model on `num_steps` batches

    model: (torch.nn.Module) the neural network
    train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches
                    training data
    metrics_save: (dict) a dictionary of functions that compute a metric using
                    the output and labels of each batch
    loss_func: the loss function
    optimizer: the optimizer
    inception: Whether the model is an inception model.
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:

        for i, train_batch in enumerate(train_loader):

            inputs,labels = train_batch
            optimizer.zero_grad()
            parallelNet = torch.nn.DataParallel(model)

            if inception:
                outputs, aux = parallelNet(tovar(inputs,requires_grad = False))
            else:
                outputs = parallelNet(tovar(inputs,requires_grad = False))

            loss = loss_func(outputs, tovar(labels, requires_grad = False))
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % save_summary_steps == 0:
                output_batch = outputs.data.cpu().numpy()
                labels_batch = labels.cpu().numpy()

               #compute all metrics on this batch
                summary_batch = {"accuracy":metrics_save["accuracy"](output_batch, labels_batch),
                                 # AUC is on binary
                                 #"AUC":metrics_save["AUC"](output_batch, labels_batch),
                                 #"mean_fpr":metrics_save["fpr"](output_batch, labels_batch),
                                 #"mean_tpr":metrics_save["tpr"](output_batch, labels_batch),
                                "loss":loss.data[0]}
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k , v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader,  metrics_save,
                        model_dir, num_epochs, loss_func,
                       optimizer, inception = False, restore_file = None):
    """
    Train the model and evaluate every epoch.

    model: (torch.nn.Module) the neural network
    train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that
                        fetches training data
    val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that
                        fetches validation data
    metrics: (dict) a dictionary of functions that compute a metric using
                        the output and labels of each batch
    model_dir: (string) directory containing config, weights and log
    restore_file: (string) optional- name of file to restore from (without its
                        extension .pth.tar)
    """

    # reload weights from restore_file if specified
   if restore_file is not None:
       restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
       logging.info("Restoring parameters from {}".format(restore_path))
       utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    best_val_auc = 0.0

    loss_func = eval(loss_func)
    optimizer = eval(optimizer)

    for epoch in range(num_epochs):

        logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, train_dataloader,metrics_save,loss_func, optimizer, inception)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader, metrics_save, loss_func)
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['AUC']

        is_best = val_acc >= best_val_acc and val_auc >= best_val_auc

        # If best_eval and auc, best_save_path
        if is_best:
            logging.info("- Found new best accuracy or auc")
            best_val_acc = val_acc
            best_val_auc = val_auc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
