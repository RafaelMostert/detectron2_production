CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/iterations_v1_100kconstantLR.yaml > outv1_constantLR.txt &
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
../configs/lofar_detection/iterations_v1_100kstepLR.yaml > outv1_stepLR.txt &
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
../configs/lofar_detection/iterations_v1_100kcosineLR.yaml > outv1_cosineLR.txt &

