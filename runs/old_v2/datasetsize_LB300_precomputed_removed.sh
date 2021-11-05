G1=0
G2=1
G3=2
G4=0
G5=1
G6=2

CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_10.yaml \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_10.txt &
sleep 10
CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_100.yaml  \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_100.txt &
sleep 40
CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_1000.yaml \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_1000.txt &
echo """
CUDA_VISIBLE_DEVICES=$G4 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_2000.yaml \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_2000.txt &
sleep 30
CUDA_VISIBLE_DEVICES=$G5 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_3000.yaml \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_3000.txt &
sleep 40
CUDA_VISIBLE_DEVICES=$G6 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_LB300_precomputed_removed_all.yaml \
    /data1/mostertrij 2 &> \
    logs/datasetsizes_LB300_precomputed_removed_all.txt &
"""
