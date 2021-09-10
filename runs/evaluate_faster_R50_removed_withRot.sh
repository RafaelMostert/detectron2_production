evaluated_at="60k"

declare -a StringArray=("uLB300_faster_R50_removed_withRot_constantLR" )

seed=1
cuda_device=0
path_to_weights="/data/mostertrij/tridentnet/output/uLB300_faster_R50_removed_withRot_constantLR_seed"$seed"/model_0059999.pth"
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../evaluate_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $path_to_weights $seed &> \
        logs/eva$evaluated_at"_"$val"_seed"$seed".txt" &
    sleep 90
done

seed=2
cuda_device=1
path_to_weights="/data/mostertrij/tridentnet/output/uLB300_faster_R50_removed_withRot_constantLR_seed"$seed"/model_0059999.pth"
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../evaluate_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $path_to_weights $seed &> \
        logs/eva$evaluated_at"_"$val"_seed"$seed".txt" &
    sleep 90
done

seed=3
cuda_device=2
path_to_weights="/data/mostertrij/tridentnet/output/uLB300_faster_R50_removed_withRot_constantLR_seed"$seed"/model_0059999.pth"
# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../evaluate_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $path_to_weights $seed &> \
        logs/eva$evaluated_at"_"$val"_seed"$seed".txt" &
    sleep 90
done
