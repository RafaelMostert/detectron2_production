# When executed, the code below will train our model using a variety of settings and implemented
# features.
val="uLB300_fast_R50_removed_withRot_constantLR"

cuda_device=0
seed=1
CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
    ../configs/lofar_detection/$val".yaml" \
    /data1/mostertrij $seed 0 100000 &> \
    logs/$val"_seed"$seed"_maxiter100000.txt" &
sleep 30

cuda_device=1
seed=2
CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
    ../configs/lofar_detection/$val".yaml" \
    /data1/mostertrij $seed 0 100000 &> \
    logs/$val"_seed"$seed"_maxiter100000.txt" &
sleep 30

cuda_device=2
seed=3
CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
    ../configs/lofar_detection/$val".yaml" \
    /data1/mostertrij $seed 0 100000 &> \
    logs/$val"_seed"$seed"_maxiter100000.txt" &
