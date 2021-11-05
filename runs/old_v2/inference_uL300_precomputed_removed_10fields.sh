CUDA_VISIBLE_DEVICES=1 python ../inference_lofar.py \
../configs/lofar_detection/inference_uL300_precomputed_removed.yaml /data1/mostertrij \
/data/mostertrij/tridentnet/output/uLB300_precomputed_removed_withRot_constantLR_seed1/model_0009999.pth &> logs/inference_uL300_precomputed_removed_10fields.txt
