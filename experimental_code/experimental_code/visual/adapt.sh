#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mail-user=ikkolasa1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --comment=adapt_office

# Load the modules required by program
module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b

# Activate the 'pytorch' environment
source activate pytorch

python tar_adaptation.py --s 2 --t 0 --max_epoch 1 --dset office-31 --net resnet50 --tag PLOT
