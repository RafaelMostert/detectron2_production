#!/bin/bash
# 1. Augment cutout
# 2. Run PyBDSF to create augmented_cat
# 3. Link sources to create retrieved_sources_cat and retrieved_components_cat
python testing_low_flux.py -o -s 321

# 4. Prepro cutouts based on augmented_cat
#-----------------------------------------------------------------------------------
#These scripts are used for the following purposes:
#
#1 - Make a list of sources from the source catalogue based on your criteria
#2 - Download the images necessary for the CLARAN cut-outs
#3 - Generating the cut-outs 

#NoteL The sample_list is created by saving a subselection of sources from the LOFAR value added catalog.
#See 'sample_list_create.py' for an example on how to do this. 
#-----------------------------------------------------------------------------------
#Set paths (change if necessary):
source /home/rafael/anaconda3/etc/profile.d/conda.sh
conda activate base
DATASET_NAME='test_low_flux_cutout'
SAMPLE_LEN=1000000000000000000000
SAMPLE_LIST='test_low_flux_cutout'
N_FIELDS=999999 # Number of fields to include. Set to 1e9 to include all fields
ROTATION=0 # 0 is False, 1 is True
single_comp_rotation_angles_deg='25,50,100'
multi_comp_rotation_angles_deg='25,50,100' #,105,125,145,165,185,205'
FIXED_CUTOUT_SIZE=1 # 0 is False, 1 is True
CUTOUT_SIZE_IN_ARCSEC=300 # Size of cut-out in DEC, will be slightly larger in RA (due to square cut-out)
RESCALED_CUTOUT_SIZE_IN_PIXELS=200
EDGE_CASES=0 # 0 is False, 1 is True
BOX_SCALE=1 #Constant with which the bounding box will be scaled
DEBUG=1 # 0 is False, 1 is True
SIG5_ISLAND_SIZE=1 # Number of pixels that need to be above 5 sigma for component to be detected
INCLUDE_LOW_SIG=0 # Determines wether low sigma sources are also labeled
ALLOW_OVERLAP=1 # Determines if we allow overlaping bounding boxes or not
INCL_DIFF=1 # Determines if difficult sources are allowed or not
DIFFICULT_LIST_NAME='difficult_1000.txt'
CLIP=1 # Determines if the cut-outs are clipped or not
CLIP_LOW=1 # lower clip value (sigma)
CLIP_HIGH=10 # upper clip value (sigma)
SIGMA_BOX_FIT=5 # Region to fit bounding box to
MUST_BE_LARGE=1 # Require sources to be > 15 arcsec or not? [0 is False, 1 is True]
MUST_BE_BRIGHT=0 # Require sources to have total flux > 10mJy or not? [0 is False, 1 is True]
UNRESOLVED_THRESHOLD=0.20
REMOVE_UNRESOLVED=0
TRAINING_MODE=0 
PRECOMPUTED_BBOXES=1 # 0 is False; 1 is True
OVERWRITE=1 # 0 is False, 1 is True

export POINTING_NAME=augmented/augmented_blurred3.82sigma_noise10sigma # Excluding the .fits extension
export CLARANPATH=/home/rafael/data/mostertrij/lofar_frcnn_tools
export IMAGEDIR=/home/rafael/data/mostertrij/data/frcnn_images # Where the folders with datasets will end up
#export IMAGEDIR=/data/mostertrij/data/frcnn_images_DR1 # Where the folders with datasets will end up
export DEBUG_PATH=/home/rafael/data/mostertrij/data/frcnn_images/$DATASET_NAME/debug # Where the folders with datasets will end up
export CACHE_PATH=/home/rafael/data/mostertrij/data/cache # Cache
export MOSAICS_PATH_DR2=/disks/paradata/shimwell/LoTSS-DR2/mosaics
export LOCAL_MOSAICS_PATH_DR2=/home/rafael/data/mostertrij/data/LoTSS_DR2
export MOSAICS_PATH_DR1=/home/rafael/data/mostertrij/pink-basics/data_LoTSS_DR1
export LOTSS_RAW_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5
export LOTSS_GAUSS_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.h5
export LOTSS_COMP_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.h5
export LOTSS_SOURCE_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.h5
export LOTSS_RAW_CATALOGUE_DR2=/home/rafael/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented/augmented_blurred3.82sigma_noise10sigma.cat.h5
#export LOTSS_RAW_CATALOGUE_DR2=/home/rafael/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented_cat_blurred0.00sigma_noise0_sigma.h5
export LOTSS_GAUSS_CATALOGUE_DR2=/home/rafael/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented/augmented_blurred3.82sigma_noise10sigma.gaus.h5
export LIKELY_UNRESOLVED_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3_DR2.csv
export OPTICAL_CATALOGUE=/home/rafael/data/mostertrij/data/catalogues/combined_panstarrs_wise.h5
export PATH=/soft/Montage_v3.3/bin:$PATH
export LOGS=/home/rafael/data/mostertrij/tridentnet/detectron2/logs
export REMOTE_IMAGES=0 # 0 is False, 1 is True
#export AUGMENTED_FLUX=1 # 0 is False, 1 is True

#1 - Make a source list: 
# Go through decision tree (fig. 5 in Williams et al 2018) and select sources that are large and
# bright
# Where n is the number of sources that you want in the list and 'list_name.fits' is the name of the 
# fits file that you want to produce. Change paths as needed. Edit script for different selection criteria.
python $CLARANPATH/imaging_scripts/multi_field_decision_tree.py $TRAINING_MODE $SAMPLE_LEN \
$SAMPLE_LIST 1 $DATASET_NAME $N_FIELDS $MUST_BE_LARGE $MUST_BE_BRIGHT #&> $LOGS/1inference_lowfluxtest.txt


#2 - Generate cut-outs using: 
#(where N and M are the index range)
#Note: The images will be placed in the directory from which you run the commands
#echo """
python -W ignore $CLARANPATH/imaging_scripts/make_cutout_frcnn.py $TRAINING_MODE $SAMPLE_LIST $SAMPLE_LEN \
    $OVERWRITE $CUTOUT_SIZE_IN_ARCSEC $RESCALED_CUTOUT_SIZE_IN_PIXELS $DATASET_NAME $ROTATION \
    $UNRESOLVED_THRESHOLD $REMOVE_UNRESOLVED #&> $LOGS/2inference_lowfluxtest.txt
#"""


#3 - Determine box size and component numbers using: 
#echo """
python $CLARANPATH/labeling_scripts/labeler_rotation.py $TRAINING_MODE $SAMPLE_LIST $OVERWRITE \
    $EDGE_CASES $BOX_SCALE $DEBUG $SIG5_ISLAND_SIZE $INCLUDE_LOW_SIG $ALLOW_OVERLAP $INCL_DIFF \
   $DATASET_NAME $single_comp_rotation_angles_deg $multi_comp_rotation_angles_deg $ROTATION \
    $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES $UNRESOLVED_THRESHOLD \
        $REMOVE_UNRESOLVED $SIGMA_BOX_FIT  #&> $LOGS/3inference_lowfluxtest.txt
#"""

#3.5 Create data split
#python $CLARANPATH/create_dataset_scripts/create_data_split.py $DATASET_NAME

#4 - Create labels in XML format and structure data in correct folder hierarchy
#echo """
python $CLARANPATH/labeling_scripts/create_and_populate_initial_dataset_rotation.py \
   $FIXED_CUTOUT_SIZE $INCL_DIFF $DIFFICULT_LIST_NAME $CLIP $CLIP_LOW $CLIP_HIGH \
   $DATASET_NAME $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES \
   $TRAINING_MODE $REMOVE_UNRESOLVED $UNRESOLVED_THRESHOLD $SIGMA_BOX_FIT #&> $LOGS/4inference_lowfluxtest.txt
#"""
#./home/rafael/data/mostertrij/lofar_frcnn_tools/create_dataset_scripts/test_lowflux.sh
conda deactivate
conda activate detectron
# 5. Run FastRcnn
# 6. Make fastcat
#eval $(conda shell.bash hook)
#conda activate detectron
evaluated_at="20k"

declare -a StringArray=("lowflux_fast_R50_withRot_constantLR" )
seed=3
cuda_device=0
path_to_weights="/data/mostertrij/tridentnet/output/uLB300_fast_R50_withRot_constantLR_seed"$seed"/model_0019999.pth"
source_cat_path="/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented/augmented_blurred3.82sigma_noise10sigma.cat.h5"
dataset_path="/data/mostertrij/data/frcnn_images/test_low_flux_cutout"
# Iterate over stringarray
echo "hey"
for val in ${StringArray[@]}; do
    echo "CUDA_VISIBLE_DEVICES=$cuda_device python general_inference.py \
        -c configs/lofar_detection/$val".yaml" \
        -s $source_cat_path  \
        -d /home/rafael/data/mostertrij -m $path_to_weights &> \
        logs/eva$evaluated_at"_"$val"_seed"$seed".txt" &"
    CUDA_VISIBLE_DEVICES=$cuda_device python general_inference.py \
        -c configs/lofar_detection/$val".yaml" \
        -s $source_cat_path \
        -da $dataset_path \
        -d /home/rafael/data/mostertrij -m $path_to_weights 
            #&> logs/eva$evaluated_at"_"$val"_seed"$seed".txt" &
    sleep 5
done
#conda deactivate


# 7. check difference fastcat and retrieved_sources_cat
# 8. report stats
python tools/compare_linked_and_predicted_cats.py \
    -l /home/rafael/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented/augmented_blurred3.82sigma_noise10sigma.linked_source_cat.h5 \
    -p /home/rafael/data/mostertrij/tridentnet/output/lowflux_fast_R50_withRot_constantLR/LoTSS_predicted_v0_merge.h5 \
    -a /home/rafael/data/mostertrij/data/LoTSS_DR2/RA13h_field/P11Hetdex12/augmented/augmented_blurred3.82sigma_noise10sigma.cat.h5 -d
