CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/precomputed_removed_iterations_v3_200kconstantLR_withRot.yaml \
/data1/mostertrij &> uitv3_removed_200kconstantLR_withRot &
#CUDA_VISIBLE_DEVICES=3 python ../evaluate_lofar.py \
#../configs/lofar_detection/precomputed_removed_iterations_v3_constantLR_withRot.yaml /data1/mostertrij \
#/data/mostertrij/tridentnet/output/v3_precomputed_removed_constantLR_withRot/model_0043999.pth &> eva_v3_precomputed_removed_constantLR_withRot &

