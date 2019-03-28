"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils as utils
import C3D_net as net
import data_loader_fc as data_loader
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help="Directory containing the dataset")
parser.add_argument('--model_dir', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

USE_CUDA = 1


def evaluate(model, val_loader, metrics_save, loss_func):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    """

    # set model to evaluation mode
    model.eval()
    #loss_func = torch.nn.BCELoss()
    # summary for current eval loop
    summ = []


    inputs, labels = val_batch

    loss = loss_func(output_batch, labels)

    # extract data from torch Variable, move to cpu, convert to numpy arrays
    output_batch = output_batch.data.cpu().numpy()
    labels_batch = labels.data.cpu().numpy()

    # compute all metrics on this batch
    summary_batch = {"accuracy":metrics_save["accuracy"](output_batch, labels_batch),
                     "AUC":metrics_save["AUC"](output_batch, labels_batch),
                     "mean_fpr":metrics_save["fpr"](output_batch, labels_batch),
                     "mean_tpr":metrics_save["tpr"](output_batch, labels_batch),
                     "loss":loss.data[0]}
    summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

def cuda(obj):
    if USE_CUDA:
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif isinstance(obj, list):
            return list(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda(device=gpus[0])
    return obj


def tovar(*arrs, **kwargs):
    tensors = [(torch.from_numpy(a) if isinstance(a, np.ndarray) else a) for a in arrs]
    vars_ = [torch.autograd.Variable(t, **kwargs) for t in tensors]
    if USE_CUDA:
        vars_ = [v.cuda() for v in vars_]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, torch.autograd.Variable) else
             v.cpu().numpy() if torch.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs




if __name__ == '__main__':
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    test_dataset = data_loader.images_dataset(data_dir = data_directory, image_list_file = test_image_list,
                                 time_tile = 1)
    test_loader = DataLoader(dataset = test_dataset, batch_size = test_batch_size,
                                shuffle = True, num_workers = 8)
    logging.info("- done.")

    # Define the model and optimizer
    model = net.SimpleCNN()
    model.cuda()

    metrics_save = net.metrics_save

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)


    # Evaluate
    test_metrics = evaluate(model, test_loader, metrics_save)
    save_path = os.path.join(model_dir, "metrics_test_{}.json".format(restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
