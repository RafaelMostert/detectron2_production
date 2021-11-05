#CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_precomputed_subsetremoved_withRot_constantLR.yaml \
#/data1/mostertrij 1 &> logs/uLB300_precomputed_subsetremoved_withRot_constantLR_1.txt &
#sleep 60
#CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_precomputed_subsetremoved_withRot_constantLR.yaml \
#/data1/mostertrij 2 &> logs/uLB300_precomputed_subsetremoved_withRot_constantLR_2.txt &
#sleep 90
#CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_precomputed_subsetremoved_withRot_constantLR.yaml \
#/data1/mostertrij 3 &> logs/uLB300_precomputed_subsetremoved_withRot_constantLR_3.txt &
CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py \
../configs/lofar_detection/LB300_Spring_60_65.yaml \
/data1/mostertrij \
/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2/model_0019999.pth \
1 &> logs/eva20k_LB300_Spring_using_uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2.txt

