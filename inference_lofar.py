#!/usr/bin/env python

print("Setup detectron2 logger")
from detectron2.utils.logger import setup_logger
setup_logger()
print("Import some common detectron2 utilities")
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, LOFARTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LOFAREvaluator

# import some common libraries
import numpy as np
from sys import argv
from cv2 import imread
import random
import os
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
random.seed(5455)

assert len(argv) > 1, "Insert path of configuration file when executing this script"
cfg = get_cfg()
cfg.merge_from_file(argv[1])
if len(argv) == 3:
    start_dir = argv[2]
    print("Beginning of paths:", start_dir)
    cfg.DATASET_PATH = cfg.DATASET_PATH.replace("/data/mostertrij",start_dir)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/data/mostertrij",start_dir)
    cfg.DATASETS.IMAGE_DIR = cfg.DATASETS.IMAGE_DIR.replace("/data/mostertrij",start_dir)
print(f"Loaded configuration file {argv[1]}")
DATASET_PATH= cfg.DATASET_PATH
print(f"Experiment: {cfg.EXPERIMENT_NAME}")
print(f"Rotation enabled: {cfg.INPUT.ROTATION_ENABLED}")
print(f"Precomputed bboxes: {cfg.MODEL.PROPOSAL_GENERATOR}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)


print("Load our data")
#def get_lofar_dicts(annotation_filepath, n_cutouts=np.inf, rotation=False):
def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    new_data = []
    counter=1
    max_value = np.inf
    if annotation_filepath.endswith('train.pkl'): 
        max_value = min(cfg.DATASETS.TRAIN_SIZE,len(dataset_dicts))
    for i in range(len(dataset_dicts)):
        if counter > max_value:
            break
        for ob in dataset_dicts[i]['annotations']:
            ob['bbox_mode'] = BoxMode.XYXY_ABS
        if cfg.MODEL.PROPOSAL_GENERATOR:
            dataset_dicts[i]["proposal_bbox_mode"] = BoxMode.XYXY_ABS
        if cfg.INPUT.ROTATION_ENABLED:
            if len(argv) == 3:
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
            new_data.append(dataset_dicts[i])
            counter+=1 
        else:
            if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
                if len(argv) == 3:
                    #dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
                    dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/home/rafael/data/mostertrij",start_dir)
                new_data.append(dataset_dicts[i])
                counter+=1
    print('len dataset is:', len(new_data), annotation_filepath)
    return new_data

# Register data inside detectron
# With DATASET_SIZES one can limit the size of these datasets
d = "inference"
DatasetCatalog.register(d, lambda d=d:
                        get_lofar_dicts(os.path.join(
                            DATASET_PATH,f"VIA_json_inference.pkl")))
MetadataCatalog.get(d).set(thing_classes=["radio_source"])

lofar_metadata = MetadataCatalog.get(d)


print("Sample and plot input data as sanity check")
inference_dict = get_lofar_dicts(os.path.join(DATASET_PATH,"VIA_json_inference.pkl")) 
#"""
for i, d in enumerate(random.sample(inference_dict, 3)):
    img = imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=lofar_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    a= vis.get_image()[:, :, ::-1]
    plt.figure(figsize=(10,10))
    plt.imshow(a)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR,f"random_input_example_for_sanity_check_{i}.jpg"))
    plt.close()
#"""


# Inference mode

# To implement the LOFAR relevant metrics I changed
# DefaultTrainer into LOFARTrainer
# where the latter calls LOFAREvaluator within build_hooks instead of the default evaluator
# this works for the after the fact test eval
# for train eval those things are somewhere within a model 
# specifically a model that takes data and retuns a dict of losses
pretrained_model_path = "/data1/mostertrij/tridentnet/output/v4_all_withRot/model_final.pth"
print("Load model:", pretrained_model_path)
cfg.MODEL.WEIGHTS = os.path.join(pretrained_model_path)  # path to the model we just trained
trainer = LOFARTrainer(cfg) 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer.resume_or_load(resume=True)

print('Load inference loader.')
inference_loader = build_detection_test_loader(cfg, f"inference")
print('Load LOFAR evaluator.')
evaluator = LOFAREvaluator(f"inference", cfg.OUTPUT_DIR, distributed=True, inference_only=True,
        kafka_to_lgm=True)
print('Start inference on dataset.')
predictions = inference_on_dataset(trainer.model, inference_loader, evaluator, overwrite=False)
print('Done with inference.')
