# When executed, the code below will train our model using a variety of settings and implemented
# features.


declare -a StringArray=("uLB300_faster_R50_removed_withRot_constantLR_IOU" )

seed=1
cuda_device=0
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" &
    sleep 90
done

seed=2
cuda_device=1
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" &
    sleep 90
done

seed=3
cuda_device=2
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" &
    sleep 90
done
