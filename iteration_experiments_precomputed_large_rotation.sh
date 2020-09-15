G1=15

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py configs/lofar_detection/precomputed_iterations_v20_101layers_rotation.yaml &
