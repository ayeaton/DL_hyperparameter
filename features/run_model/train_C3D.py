"""Train the model"""

import argparse
import logging
import os
import math
import glob
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils as utils
import C3D_net as net
import data_loader_fc as data_loader
from val_C3D import evaluate
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


def train(model,train_loader,  metrics_save, loss_func, optimizer, time_tile):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params.loss_func: the loss function
        params.optimizer: the optimizer
        params.time_tile: 0 or 1, whether there are several small videos for one video or not
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, train_batch in enumerate(train_loader):
            
            #if there are several small vidoes per video to load in (time tiling)
            if params.time_tile > 1:
                # Create a sequence of tensors to stack
                string_command = "train_batch['vid'][0,0,:,:,:]" + "," + "train_batch['vid'][0,1,:,:,:]"
                label_string = "train_batch['label'][0]" + "," + "train_batch['label'][0]"
                
                for j in range(1,train_batch['vid'].size(0)):
                    for k in range(train_batch['vid'].size(1)):
                        string_command = string_command.strip() + "," + "train_batch['vid'][%s,%s,:,:,:]"\
                        %(str(j),str(k))   
                        label_string = label_string.strip() + "," + "train_batch['label'][%s]"%str(j)  
                        
                # Stack tensors (have to reformat tensors because of the tiles)
                stacked = torch.stack((eval(string_command)), 0)

                # Format tensors
                inputs = stacked.unsqueeze(1)
                labels = torch.LongTensor(eval(label_string))
                del string_command
                del label_string
            else: 
                inputs = train_batch['vid']
                labels = train_batch['label']
            del train_batch
            
            
            # Wrap them in a Variable object
            inputs, labels = utils.tovar(inputs), utils.tovar(labels)
            
            # Get predicted Y
            parallelNet = torch.nn.DataParallel(model)
            outputs = parallelNet(inputs)
            
            #inception train has auxilary output as well 
            #outputs = model(inputs)
            
            outputs = outputs.squeeze(1)
            outputs = outputs.type(torch.FloatTensor).cuda()

            labels = labels.type(torch.FloatTensor).cuda()
               
            loss = loss_func(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = outputs.data.cpu().numpy()
                labels_batch = labels.data.cpu().numpy()
 
                
                # compute all metrics on this batch
                summary_batch = {"accuracy":metrics_save["accuracy"](output_batch, labels_batch),
                                 "AUC":metrics_save["AUC"](output_batch, labels_batch),
                                 "mean_fpr":metrics_save["fpr"](output_batch, labels_batch),
                                 "mean_tpr":metrics_save["tpr"](output_batch, labels_batch),
                                "loss":loss.data[0]}
                #summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k , v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader,  metrics_save, model_dir, num_epochs, time_tile, loss_func, optimizer, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params.model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, params.optimizer)

    best_val_acc = 0.0
    best_val_auc = 0.0
    loss_func = eval(params.loss_func)
    optimizer = eval(params.optimizer)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model = model, train_loader = train_dataloader, 
              metrics_save = metrics_save,loss_func = loss_func,
              optimizer = optimizer, time_tile = params.time_tile)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader, metrics_save, loss_func,params.time_tile)

        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['AUC']
        
        #if AUC is nan, set to 0
        if(math.isnan(val_auc)):
          val_auc = 0
            
        #if the current epoch results in the best auc and acc    
        is_best = val_acc>=best_val_acc and val_auc>=best_val_auc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=params.model_dir)

        # If best_eval and auc, best_save_path
        if is_best:
            logging.info("- Found new best accuracy or auc")
            best_val_acc = val_acc
            best_val_auc = val_auc
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(params.model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

    # Save latest val metrics in a json file in the model directory
    last_json_path = os.path.join(params.model_dir, "metrics_val_last_weights.json")
    utils.save_dict_to_json(val_metrics, last_json_path)
        

if __name__ == '__main__':
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    print("START")    
    PYTHON = sys.executable
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', 
                    help='Directory containing params.json')
    
    args = parser.parse_args()
    params = utils.Params(args.params)
    
    # pull out lr and decay for easy access
    learning_rate = params.learning_rate
    decay = params.decay

    # Set the logger
    print(os.path.join(params.model_dir, 'train.log'))
    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    
    USE_CUDA = 1
        
    # Create the input data pipeline
    logging.info("Loading the datasets...")

    train_dataset = data_loader.images_dataset(data_dir= params.data_dir, image_list_file = params.train_image_list, 
                                   time_tile = params.time_tile)
    
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = params.batch_size,
                                 shuffle = True, num_workers = 8)

    val_dataset = data_loader.images_dataset(data_dir= params.data_dir, image_list_file = params.val_image_list,
                                 time_tile = params.time_tile)
    val_loader = DataLoader(dataset = val_dataset, batch_size = params.batch_size,
                                shuffle = True, num_workers = 8)

    logging.info("- done.")


    # Define the model and optimizer
    model = net.SimpleCNN()

    logging.info("Model -- {}".format(repr(model)))

    model.cuda()

    # fetch loss function and metrics
    metrics_save = net.metrics_save

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    print(params.model_dir)

    train_and_evaluate(model = model, train_dataloader = train_loader, 
                       val_dataloader = val_loader,  metrics_save = metrics_save,
                       model_dir = params.model_dir, num_epochs = params.num_epochs, 
                       time_tile = params.time_tile, loss_func = params.loss_func, optimizer = params.optimizer)
