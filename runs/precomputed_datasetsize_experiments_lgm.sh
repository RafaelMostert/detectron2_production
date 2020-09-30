G1=0
G2=1
G3=2
G4=0
G5=1
G6=2

CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_10.yaml 1 > uitv4_10.txt &
CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_100.yaml 1 > uitv4_100.txt &
CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_1000.yaml 1 > uitv4_1000.txt &
CUDA_VISIBLE_DEVICES=$G4 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_2000.yaml 1 > uitv4_2000.txt &
CUDA_VISIBLE_DEVICES=$G5 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_3000.yaml 1 > uitv4_3000.txt &
CUDA_VISIBLE_DEVICES=$G6 python ../train_lofar.py \
../configs/lofar_detection/precomputed_datasetsizes_v4_all.yaml 1 > uitv4_all.txt &
