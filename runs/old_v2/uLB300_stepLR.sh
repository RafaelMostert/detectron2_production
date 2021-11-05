CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/uLB300_stepLR.yaml \
/data1/mostertrij 1 &> logs/uLB300_stepLR_1.txt &
sleep 60
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/uLB300_stepLR.yaml \
/data1/mostertrij 2 &> logs/uLB300_stepLR_2.txt &
sleep 90
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/uLB300_stepLR.yaml \
/data1/mostertrij 3 &> logs/uLB300_stepLR_3.txt &
#CUDA_VISIBLE_DEVICES=3 python ../train_lofar.py \
#../configs/lofar_detection/uLB300_stepLR.yaml \
#/data1/mostertrij 4 &> logs/uLB300_stepLR_4.txt &
#CUDA_VISIBLE_DEVICES=3 python ../evaluate_lofar.py \
#../configs/lofar_detection/uLB300_stepLR.yaml \
#/data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_stepLR/model_0043999.pth \
#&> logs/eva_withRot_constantLR_1.txt &

