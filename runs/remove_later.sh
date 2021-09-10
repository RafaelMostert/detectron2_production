# When executed, the code below will train our model using a variety of settings and implemented
# features.

cuda_device=0
seed=1
declare -a StringArray=("uLB300_faster_X101_removed_withRot_constantLR")
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
declare -a StringArray=("uLB300_faster_X101_removed_withRot_constantLR")
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
declare -a StringArray=("uLB300_faster_X101_removed_withRot_constantLR")
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" 
    sleep 90
done
