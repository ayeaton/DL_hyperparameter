"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import call,check_call
import sys
from sklearn.model_selection import ParameterGrid
import utils
import numpy as np

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--implementation_dir', 
                    help='Directory containing the implementation of the model')
parser.add_argument('--model_dir', 
                    help='Directory containing params.json')


#parent_dir = "/beegfs/ay1392/threedee_cnn/jupyter/"

def launch_training_job(model_dir,job_name, params, implementation_dir):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
    """
    # Create a new folder in implementation corresponding to the model
    implementation_dir = os.path.join(implementation_dir, os.path.basename(os.path.normpath(model_dir)))
    if not os.path.exists(implementation_dir):
        os.makedirs(implementation_dir)
        
    implementation_hyperparams_dir = os.path.join(implementation_dir, job_name)
    if not os.path.exists(implementation_hyperparams_dir):
        os.makedirs(implementation_hyperparams_dir)
        
    params.implementation_dir = implementation_hyperparams_dir + "/"
    
    # Write parameters in json file
    json_path = os.path.join(implementation_hyperparams_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} {model_dir}/train_C3D.py --params={json_path}".format(python=PYTHON, model_dir=model_dir, json_path=json_path)
    #print(cmd)
    

    #NOT GENERALIZABLE -- READ IN TEMPLATE AND APPEND?
    f = open(os.path.join(implementation_hyperparams_dir, ('run_' + job_name + '.test')), 'w+')
    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("#SBATCH --job-name=iterate{}\n".format(job_name))
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --mem=100GB\n") 
    f.write("#SBATCH --time=12:00:00\n")
    f.write("#SBATCH --gres=gpu:1 -c1\n")
    f.write("#SBATCH --cpus-per-task=1\n")
    f.write("#SBATCH --error={}.out\n".format(model_dir + "/" + job_name))
    f.write("\n")
    f.write("\n")
    f.write("module purge\n")
    f.write("module load python3/intel/3.5.3\n")
    f.write("module load pillow/intel/4.0.0\n")
    f.write("module load scikit-learn/intel/0.18.1\n")
    f.write("module load pytorch/python3.5/0.2.0_3\n")
    f.write("module load numpy/intel/1.13.1 \n")
    f.write("module load cuda/8.0.44\n")
    f.write("module load jupyter-kernels/py3.5\n")
    f.write("module load mysql/5.7.17\n")
    f.write("module load zeromq/intel/4.2.0\n")
    f.write("module load intel/17.0.1\n")
    f.write("module load zlib/intel/1.2.8\n")
    f.write("\n")
    f.write("\n")
    f.write(cmd)
    f.close()

    file=(implementation_hyperparams_dir +'/run_' + job_name + '.test')
    sbatch_call = "sbatch " + file
    print(sbatch_call)
    call(sbatch_call, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file

    arg = parser.parse_args()
    json_path = os.path.join(arg.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch 
    param_grid = {'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'decay': [0,1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'batch_size':[5]}
    
    
    grid = ParameterGrid(param_grid)
    
    grid_chose = np.random.choice(len(list(grid)), 1)
    
    
    for i in grid_chose:
        learning_rate = grid[i]["learning_rate"]
        decay = grid[i]["decay"]
        batch_size = grid[i]["batch_size"]
        
        params.learning_rate = learning_rate
        params.decay = decay
        params.batch_size = batch_size
        
        job_name = "learning_rate_{}decay_{}batch_{}".format(learning_rate, decay, batch_size)
        launch_training_job(arg.model_dir, job_name, params, arg.implementation_dir)


