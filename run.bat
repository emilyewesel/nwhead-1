#!/bin/bash
#SBATCH --job-name nw_edema
#SBATCH --time 22:00:00
#SBATCH --gres gpu:titanrtx:1
#SBATCH -c 4
#SBATCH --mem 32gb
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user magdapaschali@gmail.com

conda activate support_set
python3 train.py --train_method nwhead --train_class Edema --optimizer sgd --arch densenet121 --wandb_kwargs name=nw_edema_densenet project=support_set_bias entity=magda --dataset chexpert  --wandb_api_key_path /dataNAS/people/paschali/git/nwhead-1/key_file.txt
