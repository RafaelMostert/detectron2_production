# When executed, the code below will train our model using a variety of settings and implemented
# features.

cuda_device=0
seed=1


declare -a StringArray=("uLB300_fast_R50_removed_withRot_constantLR" 
"Fedora" "Red Hat Linux" 
"Ubuntu" "Debian" )

# Iterate over stringarray
for val in ${StringArray[@]}; do
    python test.py \
        ../configs/lofar_detection/$val".yaml" \
        /data1/mostertrij $seed 
    echo sleep 90

done
