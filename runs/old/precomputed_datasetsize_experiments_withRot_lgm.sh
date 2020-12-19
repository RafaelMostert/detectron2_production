G1=1
G2=2
G3=3
G4=0
G5=1
G6=2

#CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py \
#../configs/lofar_detection/precomputed_datasetsizes_v4_10_withRot.yaml 1 > uitv4_10_withRot.txt &
#CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py \
#../configs/lofar_detection/precomputed_datasetsizes_v4_100_withRot.yaml 1 > uitv4_100_withRot.txt &
#CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py \
#../configs/lofar_detection/precomputed_datasetsizes_v4_1000_withRot.yaml 1 > uitv4_1000_withRot.txt &
CUDA_VISIBLE_DEVICES=$G4 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_2000_withRot.yaml 1 > uitv4_2000_withRot.txt &
CUDA_VISIBLE_DEVICES=$G5 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_3000_withRot.yaml 1 > uitv4_3000_withRot.txt &
CUDA_VISIBLE_DEVICES=$G6 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_all_withRot.yaml 1 > uitv4_all_withRot.txt &
