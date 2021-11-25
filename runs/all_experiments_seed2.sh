# When executed, the code below will train our model using a variety of settings and implemented
# features.

cuda_device=1
seed=2

declare -a StringArray=( #"uLB300_fast_R50_removed_withRot_constantLR" "uLB300_fast_X50_removed_withRot_constantLR"
#"uLB300_fast_R101_removed_withRot_constantLR" "uLB300_fast_X101_removed_withRot_constantLR"
#"uLB300_fast_R50_removed_withRot_stepLR" "uLB300_fast_R50_removed_withRot_cosineLR"
#"uLB300_fast_R50_withRot_constantLR" "uLB300_fast_R50_removed_constantLR"
"uLB300_fast_R50_constantLR" )

# Iterate over stringarray
for val in ${StringArray[@]}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" 
    sleep 90
done
