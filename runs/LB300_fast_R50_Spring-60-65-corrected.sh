CUDA_VISIBLE_DEVICES=2 python ../evaluate_lofar.py \
../configs/lofar_detection/LB300_fast_R50_Spring_60_65_corrected.yaml \
/data1/mostertrij \
/data/mostertrij/tridentnet/output/uLB300_fast_R50_removed_withRot_constantLR_seed3_maxiter100000/model_0019999.pth \
3 &> logs/eva20k_LB300_Spring_using_uLB300_fast_R50_removed_withRot_constantLR_seed3_maxiter100000.txt

#CUDA_VISIBLE_DEVICES=0 python ../inference_lofar.py \
#../configs/lofar_detection/LB300_Spring_60_65_corrected.yaml /data1/mostertrij \
#/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2/model_0019999.pth \
#&> logs/inf20k_LB300_Spring_using_uLB300_precomputed_removed_withRot_constantLR_testIOU_seed2.txt
#../configs/lofar_detection/inference_LB300_Spring_60_65_corrected.yaml /data1/mostertrij \
