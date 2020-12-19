#!/bin/bash
#SBATCH -p gpu-long
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
         
python ../train_lofar.py ../configs/lofar_detection/iterations_v12_100kstepLR.yaml
