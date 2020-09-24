#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
         
python projects/TridentNet/lofar_evaluate.py /home/mostertrij/detectron2/projects/TridentNet/configs/v12_norot_tridentnet_400imsize_fast_R_101_C4_3x.yaml &> v12_eval.txt
