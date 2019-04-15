#!/bin/bash

#SBATCH --job-name=submit_iterate
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --error=iterate


module purge
module load python3/intel/3.5.3
module load pillow/intel/4.0.0
module load scikit-learn/intel/0.18.1
module load pytorch/python3.5/0.2.0_3
module load numpy/intel/1.13.1 
module load cuda/8.0.44
module load jupyter-kernels/py3.5
module load mysql/5.7.17
module load zeromq/intel/4.2.0
module load intel/17.0.1
module load zlib/intel/1.2.8


/share/apps/python3/3.5.3/intel/bin/python3.5 /beegfs/ay1392/Embryo_models/models/C3D_dartmouth/Iterate.py \
        --implementation_dir="/beegfs/ay1392/Embryo_models/implementations/C3D_dartmouth" \
        --model_dir="/beegfs/ay1392/Embryo_models/models/C3D_dartmouth"


