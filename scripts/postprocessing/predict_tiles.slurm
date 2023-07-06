#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=output_predict_tiles_%A.out

conda activate /projects/0/einf512/miniconda3/envs/main/
srun python ~/postprocessing/predict_tiles_from_cluster.py model_1684233861_L2_w20_k5_f16_a20/encoder_epoch_9
