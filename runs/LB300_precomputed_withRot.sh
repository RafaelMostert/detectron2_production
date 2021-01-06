CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/LB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 1 &> logs/LB300_precomputed_withRot_constantLR_1.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/LB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 2 &> logs/LB300_precomputed_withRot_constantLR_2.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/LB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 3 &> logs/LB300_precomputed_withRot_constantLR_3.txt &
#sleep 30
#CUDA_VISIBLE_DEVICES=3 python ../train_lofar.py \
#../configs/lofar_detection/LB300_precomputed_withRot_constantLR.yaml \
#/data1/mostertrij 4 &> logs/LB300_precomputed_withRot_constantLR_4.txt &
#CUDA_VISIBLE_DEVICES=3 python ../evaluate_lofar.py \
#../configs/lofar_detection/LB300_precomputed_withRot_constantLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/LB300_precomputed_withRot_constantLR/model_0043999.pth \
#&> logs/eva_precomputed_withRot_constantLR_1.txt &

