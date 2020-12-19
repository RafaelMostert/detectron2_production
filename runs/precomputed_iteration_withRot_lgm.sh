#CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
#../configs/lofar_detection/precomputed_iterations_v3_constantLR_withRot.yaml /data1/mostertrij &> uitv3_precomputed_constantLR_withRot &
CUDA_VISIBLE_DEVICES=2 python ../evaluate_lofar.py \
../configs/lofar_detection/precomputed_iterations_v3_constantLR_withRot.yaml /data1/mostertrij \
/data/mostertrij/tridentnet/output/v3_precomputed_constantLR_withRot/model_0039999.pth &> eva_v3_precomputed_constantLR_withRot &

