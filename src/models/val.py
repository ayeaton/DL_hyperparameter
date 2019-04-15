"""Evaluates the model"""

#################################################################################
#   val.py from CS230 Deep Learning at Stanford
#     TAs: Surag Nair, Guilluame Genthial, Olivier Moindrot
#     https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
################################################################################
from comet_ml import Experiment
import argparse
import logging
import os
import numpy as np
import sys
sys.path.append("/home/ay1392/anna_beegfs/pytorch_projects/features_proj/src")

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import utils.utils as utils
import utils.metrics_code as metrics_code


USE_CUDA = 1

def evaluate(model, val_loader, metrics_save, loss_func, experiment = None, inception = False):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    with experiment.test():
        for i, val_batch in enumerate(val_loader):

            inputs,labels = val_batch
            labels = labels.type(torch.LongTensor)

            parallelNet = torch.nn.DataParallel(model)
            outputs = parallelNet(utils.tovar(inputs,requires_grad = False))
            #outputs = model(utils.tovar(inputs))

            loss = loss_func(outputs, utils.tovar(labels))

            if inception:
                softmax = nn.Softmax()
                outputs = softmax(outputs)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = outputs.data.cpu().numpy()
            labels_batch = labels.data.cpu().numpy()

            ypred = []
            for j, x in enumerate(output_batch):
                ypred.append(x[labels_batch[j]])
            logging.info(ypred)

            # compute all metrics on this batch
            summary_batch = {"test_accuracy":metrics_save["accuracy"](ypred, labels_batch),
                             "test_AUC":metrics_save["AUC"](ypred, labels_batch),
                             "test_mean_fpr":metrics_save["fpr"](ypred, labels_batch),
                             "test_mean_tpr":metrics_save["tpr"](ypred, labels_batch),
                             "test_loss":loss.data[0]}
            experiment.log_metrics(summary_batch, step=i)
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def main():
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help="Directory containing the dataset")
    parser.add_argument('--model_dir', help="Directory containing params.json")
    parser.add_argument('--params',
                        help='Directory containing params.json')
    parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                         containing weights to load")

    params = utils.Params(args.params)

    # Get the logger
    utils.set_logger(os.path.join(params.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    test_dataset = dataset(file_path = params.metadata_file, split = "Test",
                        classes = params.classes)

    test_loader = DataLoader(dataset = test_dataset, batch_size = params.batch_size,
                                 shuffle = True, num_workers = 8)

    logging.info("- done.")

    # Define the model and optimizer
    if model != "Inception":
        net = importlib.import_module("features.models.{}".format(params.model))
        model = net.Net()
        inception = False
    else:
        model = models.inception_v3(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        model.AuxLogits.fc = nn.Linear(768, 1)
        inception = True

    model.cuda()

    metrics_save = metrics_code.metrics_save

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)


    # Evaluate
    test_metrics = evaluate(model, test_loader, metrics_save, experiment, inception)
    save_path = os.path.join(model_dir, "metrics_test_{}.json".format(restore_file))
    utils.save_dict_to_json(test_metrics, save_path)


if __name__ == '__main__':
    main()
