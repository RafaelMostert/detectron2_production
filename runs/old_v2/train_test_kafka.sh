#python ../train_lofar.py ../configs/lofar_detection/training_1field_kafka.yaml /home/rafael/data/mostertrij
python ../evaluate_lofar.py ../configs/lofar_detection/training_1field_kafka.yaml /home/rafael/data/mostertrij \
/data/mostertrij/tridentnet/output/training_1field_kafka/model_final.pth
#python -m cprofilev ../train_lofar.py ../configs/lofar_detection/training_1field_kafka.yaml /home/rafael/data/mostertrij
