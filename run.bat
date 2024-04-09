#!/bin/bash
#SBATCH --job-name nw_cardio_0.0001_adamw_resnet18_no_pre_n_shot_8
#SBATCH --time 12:00:00
#SBATCH --gres gpu:titanrtx:1
#SBATCH -c 4
#SBATCH --mem 32gb
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user magdapaschali@gmail.com

conda activate support_set
python3 train.py --train_method nwhead --n_shot 8 --train_class Cardiomegaly --optimizer adamw --arch resnet18 --no_pretrained --lr 1e-4 --wandb_kwargs name=nw_cardio_8_0.0001_no_pretrained_resnet18_adamw project=support_set_bias entity=magda --dataset chexpert --wandb_api_key_path /dataNAS/people/paschali/git/nwhead-1/key_file.txt
