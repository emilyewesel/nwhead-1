#!/bin/bash
#SBATCH --job-name=classify_non_gated_ct_scans
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --gpus 3
#SBATCH -c 20
#SBATCH --mail-type=BEGIN,END,FAIL

python3 train.py --dataset chexpert