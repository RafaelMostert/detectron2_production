# When executed, the code below will train our model using a variety of settings and implemented
# features.

val="uLB300_faster_R50_removed_withRot_constantLR"

seed=1
CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" & 
sleep 90
seed=2
CUDA_VISIBLE_DEVICES=1 python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" & 
sleep 90
seed=3
CUDA_VISIBLE_DEVICES=2 python ../train_lofar.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed &> \
        logs/$val"_seed"$seed".txt" 
