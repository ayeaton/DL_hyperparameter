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

import data_loader_fc as data_loader
from val_C3D import evaluate
import feature.utils.metrics as metrics
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from comet_ml import Experiment


def train(model, train_loader,  metrics_save, loss_func, optimizer):
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

        for i, train_batch in enumerate(train_loader, 0):

            inputs, labels = train_batch
            optimizer.zero_grad()
            outputs = parallelNet(inputs)
            loss = loss_func(outputs, labels)
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


def train_and_evaluate(model, train_dataloader, val_dataloader,  metrics_save, model_dir, num_epochs, loss_func, optimizer, restore_file=None):
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
              optimizer = optimizer)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader, metrics_save, loss_func)

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

    experiment = Experiment(api_key="ysBHHTJLIBAhlklc4sdBd0vlp",
                        project_name="general", workspace="ayeaton")

# do i need this
    PYTHON = sys.executable
    # add the params argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',
                    help='Directory containing params.json')
    # parse the args
    args = parser.parse_args()
    # loads hyperparams from a json file
    params = utils.Params(args.params)
    # Set the logger
    print(os.path.join(params.model_dir, 'train.log'))
    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    # Create the input data pipeline
    logging.info("Loading the datasets...")

    USE_CUDA = 1

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    logging.info("- done.")

    #import model specified in params file
    import params.model as net

    # Define the model and optimizer
    # all models are called Net
    model = net.Net()

    logging.info("Model -- {}".format(repr(model)))

    model.cuda()

    # fetch loss function and metrics
    metrics_save = metrics.metrics_save

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    print(params.model_dir)

    train_and_evaluate(model = model, train_dataloader = train_loader,
                       val_dataloader = val_loader,  metrics_save = metrics_save,
                       model_dir = params.model_dir, num_epochs = params.num_epochs,
                        loss_func = params.loss_func, optimizer = params.optimizer)
