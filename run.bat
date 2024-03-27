#!/bin/bash
#SBATCH --job-name fc_cardio_0.0001_adamw_resnet18
#SBATCH --time 12:00:00
#SBATCH --gres gpu:2080ti:1
#SBATCH -c 4
#SBATCH --mem 32gb
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user magdapaschali@gmail.com

conda activate support_set
python3 train.py --train_method fchead --train_class Cardiomegaly --optimizer adamw --arch resnet18 --lr 1e-4 --wandb_kwargs name=fc_cardio_0.0001_pretrained_resnet18_adamw_no_pretraining project=support_set_bias entity=magda --dataset chexpert --wandb_api_key_path /dataNAS/people/paschali/git/nwhead-1/key_file.txt
