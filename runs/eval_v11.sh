#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
         
python projects/TridentNet/lofar_evaluate.py /home/mostertrij/detectron2/projects/TridentNet/configs/v11_norot_tridentnet_fast_R_101_C4_3x.yaml &> v11_eval2.txt
