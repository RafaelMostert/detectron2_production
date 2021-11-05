G1=0
G2=1
G3=2
G4=0
G5=1
G6=2
seed=1

echo """
CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml \
    /data1/mostertrij $seed 10 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_10.txt" &
sleep 10
CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml  \
    /data1/mostertrij $seed 100 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_100.txt" &
sleep 40
CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml \
    /data1/mostertrij $seed 1000 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_1000.txt" &
"""
CUDA_VISIBLE_DEVICES=$G4 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml \
    /data1/mostertrij $seed 2000 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_2000.txt" &
sleep 30
CUDA_VISIBLE_DEVICES=$G5 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml \
    /data1/mostertrij $seed 3000 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_3000.txt" &
sleep 40
CUDA_VISIBLE_DEVICES=$G6 python ../train_lofar.py \
    ../configs/lofar_detection/datasetsizes_uLB300_precomputed_removed.yaml \
    /data1/mostertrij $seed 4000 &> \
    logs/datasetsizes_uLB300_precomputed_removed_seed$seed"_all.txt" &
