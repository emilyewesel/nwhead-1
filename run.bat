#!/bin/bash
#SBATCH --job-name nw_head_cardio
#SBATCH --time 10:00:00
#SBATCH --gres gpu:2080ti:1
#SBATCH -c 48
#SBATCH --mem 64gb
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user magdapaschali@gmail.com

conda activate support_set
python3 train.py --dataset chexpert --wandb_api_key_path /dataNAS/people/paschali/git/nwhead-1/key_file.txt --wandb_kwargs name=nwhead_cardio project=support_set_bias