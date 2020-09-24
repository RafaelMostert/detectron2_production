#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
         
python /home/mostertrij/detectron2/train_lofar.py /home/mostertrij/detectron2/configs/lofar_detection/iterations_v10_100kconstantLR.yaml
