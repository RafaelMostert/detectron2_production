G1=0
G2=1
G3=2

#CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_10.yaml 1 > uit_v2_10.txt &
#CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_100.yaml 1 > uit_v2_100.txt &
#CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_1000.yaml 1 > uit_v2_1000.txt &
CUDA_VISIBLE_DEVICES=$G1 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_2000.yaml 1 > uit_v2_2000.txt &
CUDA_VISIBLE_DEVICES=$G2 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_3000.yaml 1 > uit_v2_3000.txt &
CUDA_VISIBLE_DEVICES=$G3 python ../train_lofar.py ../configs/lofar_detection/datasetsizes_v2_all.yaml 1 > uit_v2_all.txt &
