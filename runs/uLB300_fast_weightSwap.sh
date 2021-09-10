# When executed, the code below will train our model using a variety of settings and implemented
# features.
declare -a StringArray=("uLB300_fast_R50_removed_withRot_constantLR_initializeWeights")

cuda_device=0
seed=1
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" & 
    sleep 90
done

cuda_device=1
seed=2
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" & 
    sleep 90
done

cuda_device=2
seed=3
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" & 
    sleep 90
done
