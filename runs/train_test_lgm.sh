CUDA_VISIBLE_DEVICES=0 python ../train_lofar.py \
../configs/lofar_detection/training_1field_kafka.yaml /data1/mostertrij
CUDA_VISIBLE_DEVICES=0 python ../evaluate_lofar.py ../configs/lofar_detection/training_1field_kafka.yaml /data1/mostertrij \
/data/mostertrij/tridentnet/output/training_1field_kafka/model_final.pth
#python -m cprofilev ../train_lofar.py ../configs/lofar_detection/training_1field_kafka.yaml /home/rafael/data/mostertrij
