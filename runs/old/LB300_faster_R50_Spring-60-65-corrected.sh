CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py \
../configs/lofar_detection/LB300_faster_R50_Spring_60_65_corrected.yaml \
/data1/mostertrij \
/data/mostertrij/tridentnet/output/uLB300_faster_R50_removed_withRot_constantLR_seed1/model_0059999.pth \
1 &> logs/eva60k_LB300_Spring_using_uLB300_faster_R50_removed_withRot_constantLR_seed1.txt

#CUDA_VISIBLE_DEVICES=0 python ../inference_lofar.py \
#../configs/lofar_detection/LB300_Spring_60_65_corrected.yaml /data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2/model_0019999.pth \
#&> logs/inf20k_LB300_Spring_using_uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2.txt
#../configs/lofar_detection/inference_LB300_Spring_60_65_corrected.yaml /data1/mostertrij \
