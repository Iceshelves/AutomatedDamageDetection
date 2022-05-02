#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18 

srun python ~/preprocessing/scripts/train-vae/main.py ~/preprocessing/scripts/train-vae/train-vae.ini


