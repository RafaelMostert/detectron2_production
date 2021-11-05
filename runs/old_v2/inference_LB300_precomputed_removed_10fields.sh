CUDA_VISIBLE_DEVICES=0 python ../inference_lofar.py \
../configs/lofar_detection/inference_LB300_precomputed_removed.yaml /data1/mostertrij &> logs/inference_10fields.txt
