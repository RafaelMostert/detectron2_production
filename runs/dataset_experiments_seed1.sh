# When executed, the code below will train our model using a variety of settings and implemented
# features.

cuda_device=0
seed=1

declare -a StringArray=( #"uLB300_fast_R50_removed_withRot_constantLR"
 #"uLB300_fast_R50_withRot_constantLR" "uLB300_fast_R50_removed_constantLR"
"uLB300_fast_R50_constantLR" )

declare -a iterations=( 100 1000 2000 3000 )

# Iterate over stringarray
for val in ${StringArray[@]}; do
    for dat in ${iterations[@]}; do

        CUDA_VISIBLE_DEVICES=$cuda_device python ../train_lofar.py \
            ../configs/lofar_detection/$val"_trainsize.yaml" \
            /data1/mostertrij $seed $dat &> \
            logs/$val"_trainsize"$dat"_seed"$seed".txt" 
        sleep 90
    done

done
