#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:4
         
python projects/TridentNet/lofar_train_tridentnet.py --resume --num-gpus=4 --config-file /home/mostertrij/detectron2/projects/TridentNet/configs/v11_norot_tridentnet_fast_R_101_C4_3x.yaml
