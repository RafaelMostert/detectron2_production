#CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
#../configs/lofar_detection/LB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
#/data1/mostertrij 1 &> logs/LB300_precomputed_removed_withRot_constantLR_1_threshold0_20.txt &
#sleep 60
#CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
#../configs/lofar_detection/LB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
#/data1/mostertrij 2 &> logs/LB300_precomputed_removed_withRot_constantLR_2_threshold0_20.txt &
#sleep 90
#CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
#../configs/lofar_detection/LB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
#/data1/mostertrij 3 &> logs/LB300_precomputed_removed_withRot_constantLR_3_threshold0_20.txt &
CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py \
../configs/lofar_detection/LB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij \
/data/mostertrij/tridentnet/output/LB300_precomputed_removed_withRot_constantLR_seed1/model_0029999.pth \
1 &> logs/eva30k_precomputed_removed_withRot_constantLR_1.txt &

