#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=gpu
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --mail-user=ikkolasa1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --comment=pretrain__office31

# Load the modules required by program
module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b

# Activate the 'pytorch' environment
source activate pytorch

python src_pretrain.py --dset office-31 --s 0 --t 1 --mode train+test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 0 --t 2 --mode test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 1 --t 0 --mode train+test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 1 --t 2 --mode test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 2 --t 0 --mode train+test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 2 --t 1 --mode test --trte full --seed 2020 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 0 --t 1 --mode train+test --trte full --seed 2021 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 0 --t 2 --mode test --trte full --seed 2021 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 1 --t 0 --mode train+test --trte full --seed 2021 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 1 --t 2 --mode test --trte full --seed 2021 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 2 --t 0 --mode train+test --trte full --seed 2021 --max_epoch 40 --net resnet50

python src_pretrain.py --dset office-31 --s 2 --t 1 --mode test --trte full --seed 2021 --max_epoch 40 --net resnet50
