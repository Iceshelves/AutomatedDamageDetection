#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18 
#SBATCH --mem=50G

# RUN SCRIPT WITH THE FOLLOWING COMMAND TO HAVE file.ini STORED WITH JOB-ID
# RES=$(sbatch --parsable ../../preprocessing/scripts/train-vae/train.slurm) && echo $RES && cp ../../preprocessing/scripts/train-vae/train-vae.ini ./train-vae-$RES.ini

#conda activate /projects/0/einf512/miniconda3/envs/main/
conda activate /gpfs/home3/mizeboud/mambaforge/envs/jupyter_dask
srun python ~/preprocessing/scripts/train-vae/main.py ~/preprocessing/scripts/train-vae/train-vae.ini