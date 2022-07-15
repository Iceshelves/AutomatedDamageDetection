#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=3 # You could request 3 GPUs (--gpus-per-node=3) without increasing the price
#SBATCH --cpus-per-task=18 
#SBATCH --mem=300G

srun python ~/preprocessing/scripts/train-vae/main.py ~/preprocessing/scripts/train-vae/train-vae.ini


