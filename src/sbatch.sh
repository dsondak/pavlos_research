#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-08:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nhoernle@g.harvard.edu

# source new-modules.sh

module load python/3.6.0-fasrc01
#module load gcc/5.2.0-fasrc02
# module load cuda/8.0-fasrc01
#module load cuda
# module load cudnn/5.1-fasrc02
#module load theano/20160126-fasrc01
#module load Anaconda3/4.3.0-fasrc01

## do once
#echo "provisioning"
#conda create -n thesisenv2 --clone="$PYTHON_HOME"
#pip install pystan
#conda install -c conda-forge theano  -y
#conda install -c conda-forge libgpuarray -y
#conda install -c conda-forge joblib -y
#conda install -c conda-forge pygpu -y
#conda install tqdm -y
#conda install -c conda-forge pymc3 -y
#echo "done provisioning"

source activate essil
./dispatch_script.py 'cw_stan.pkl' $1 $2 $3 $4 $5
source deactivate essil

echo "finished"
