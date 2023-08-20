#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=gpu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --mail-user=ikkolasa1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --comment=adapt_ppa

# Load the modules required by program
module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b

# Activate the 'pytorch' environment
source activate ogbg

python tar_adaptation.py