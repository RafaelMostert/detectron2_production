CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 1 &> logs/uLB300_precomputed_removed_withRot_constantLR_1_threshold0_20.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 2 &> logs/uLB300_precomputed_removed_withRot_constantLR_2_threshold0_20.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 3 &> logs/uLB300_precomputed_removed_withRot_constantLR_3_threshold0_20.txt
#CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_precomputed_removed_withRot_constantLR_threshold0.20.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_seed1/model_0009999.pth \
#1 #&> logs/eva10k_uLB300_precomputed_removed_withRot_constantLR_1.txt
sleep 180

CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 1 &> logs/uLB300_precomputed_withRot_constantLR_1.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 2 &> logs/uLB300_precomputed_withRot_constantLR_2.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_precomputed_withRot_constantLR.yaml \
/data1/mostertrij 3 &> logs/uLB300_precomputed_withRot_constantLR_3.txt
#CUDA_VISIBLE_DEVICES=1 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_precomputed_withRot_constantLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_seed3/model_0019999.pth 3 \
#&> logs/eva_precomputed_withRot_constantLR_3.txt &
sleep 180

CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 1 &> logs/uLB300_removed_withRot_constantLR_1_threshold0_20.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 2 &> logs/uLB300_removed_withRot_constantLR_2_threshold0_20.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_removed_withRot_constantLR_threshold0.20.yaml \
/data1/mostertrij 3 &> logs/uLB300_removed_withRot_constantLR_3_threshold0_20.txt
#CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_removed_withRot_constantLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_removed_withRot_constantLR_seed2/model_0199999.pth \
#2 &> logs/eva200k_removed_withRot_constantLR_2.txt &
sleep 180

CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_withRot_constantLR.yaml \
/data1/mostertrij 1 &> logs/uLB300_withRot_constantLR_1.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_withRot_constantLR.yaml \
/data1/mostertrij 2 &> logs/uLB300_withRot_constantLR_2.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_withRot_constantLR.yaml \
/data1/mostertrij 3 &> logs/uLB300_withRot_constantLR_3.txt
#CUDA_VISIBLE_DEVICES=3 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_withRot_constantLR.yaml \
#/data1/mostertrij 4 &> logs/uLB300_withRot_constantLR_4.txt &
#CUDA_VISIBLE_DEVICES=3 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_withRot_constantLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_withRot_constantLR/model_0043999.pth \
#&> logs/eva_withRot_constantLR_1.txt &
sleep 180

CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_constantLR.yaml \
/data1/mostertrij 1 &> logs/uLB300_constantLR_1.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_constantLR.yaml \
/data1/mostertrij 2 &> logs/uLB300_constantLR_2.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_constantLR.yaml \
/data1/mostertrij 3 &> logs/uLB300_constantLR_3.txt
#CUDA_VISIBLE_DEVICES=3 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_constantLR.yaml \
#/data1/mostertrij 4 &> logs/uLB300_constantLR_4.txt &
#CUDA_VISIBLE_DEVICES=3 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_constantLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_constantLR/model_0043999.pth \
#&> logs/eva_withRot_constantLR_1.txt &
