#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
         
python ../train_lofar.py ../configs/lofar_detection/iterations_v10_300kconstantLR.yaml
