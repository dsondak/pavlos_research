#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-08:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pblankley@g.harvard.edu

source new-modules.sh

module load python/3.6.0-fasrc01
#module load gcc/5.2.0-fasrc02
# module load cuda/8.0-fasrc01
#module load cuda
# module load cudnn/5.1-fasrc02
#module load theano/20160126-fasrc01
module load Anaconda3/4.3.0-fasrc01

## do once
echo "provisioning"
conda create -n pvsearch --clone="$PYTHON_HOME"
conda install tqdm -y
pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install pandas
pip install scipy
pip install matplotlib
pip install seaborn
echo "done provisioning"

source activate pvsearch
./run_experiments.py  $1 $2 $3 $4 $5
source deactivate pvsearch

echo "finished"
