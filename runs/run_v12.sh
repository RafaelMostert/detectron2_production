#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --ntasks=1
#SBATCH --time=23:55:00
#SBATCH --gres=gpu:4
         
python projects/TridentNet/lofar_train_tridentnet.py --num-gpus=4 --config-file /home/mostertrij/detectron2/projects/TridentNet/configs/v12_norot_tridentnet_400imsize_fast_R_101_C4_3x.yaml
