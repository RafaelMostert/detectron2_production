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
import pandas as pd
from collections import OrderedDict
from astropy.table import Table
from copy import deepcopy
#from separation import separation
from astropy.coordinates import SkyCoord
import astropy.units as u
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
random.seed(5455)

assert len(argv) == 4, ("Insert 1) path of configuration file, 2) beginning of file path, 3) full "
    "path to model weights, when executing this script")
cfg = get_cfg()
cfg.merge_from_file(argv[1])
lotss_dr2_path = '/data/mostertrij/data/catalogues/LoTSS_DR2_v100.srl.h5'

# Adjust beginning of file paths
start_dir = argv[2]
print("Beginning of paths:", start_dir)
cfg.DATASET_PATH = cfg.DATASET_PATH.replace("/data/mostertrij",start_dir)
cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/data/mostertrij",start_dir)
cfg.DATASETS.IMAGE_DIR = cfg.DATASETS.IMAGE_DIR.replace("/data/mostertrij",start_dir)
lotss_dr2_path = lotss_dr2_path.replace("/data/mostertrij",start_dir)

# Read in path to weights that will be evaluated on its own train/test/val
# For example: pretrained_model_path = "/data/mostertrij/tridentnet/output/v3_precomputed_constantLR_withRot_no_box_reg/model_0005999.pth"
pretrained_model_path = argv[3].replace('/data/mostertrij',start_dir)

assert os.path.exists(lotss_dr2_path), lotss_dr2_path
print(f"Loaded configuration file {argv[1]}")
DATASET_PATH= cfg.DATASET_PATH
print(f"Experiment: {cfg.EXPERIMENT_NAME}")
print(f"Rotation enabled: {cfg.INPUT.ROTATION_ENABLED}")
print(f"Precomputed bboxes: {cfg.MODEL.PROPOSAL_GENERATOR}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
print("Pretrained model path:", pretrained_model_path)
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

        if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
            if len(argv) == 4:
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data2/mostertrij",start_dir)
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
            new_data.append(dataset_dicts[i])
            counter+=1

    print('len dataset is:', len(new_data), annotation_filepath)
    return new_data

# Register data inside detectron
for d in ["train", "val", "test"]:
    DatasetCatalog.register(d, 
                            lambda d=d:
                            get_lofar_dicts(os.path.join(
                                DATASET_PATH,f"VIA_json_{d}.pkl")))
    MetadataCatalog.get(d).set(thing_classes=["radio_source"])
lofar_metadata = MetadataCatalog.get("train")

# Inference mode

# To implement the LOFAR relevant metrics I changed
# DefaultTrainer into LOFARTrainer
# where the latter calls LOFAREvaluator within build_hooks instead of the default evaluator
# this works for the after the fact test eval
# for train eval those things are somewhere within a model 
# specifically a model that takes data and retuns a dict of losses
print("Load model:", pretrained_model_path)
cfg.MODEL.WEIGHTS = os.path.join(pretrained_model_path)  # path to the model we just trained
trainer = LOFARTrainer(cfg) 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer.resume_or_load(resume=True)

for d in ["train", "val", "test"]:
    print(f"For {d} set:")
    print('Load inference loader.')
    inference_loader = build_detection_test_loader(cfg, d)
    print('Load LOFAR evaluator.')
    evaluator = LOFAREvaluator(d, cfg.OUTPUT_DIR, distributed=True, inference_only=False,
            save_predictions=True, kafka_to_lgm=False,component_save_name="bare_predicted_component_catalogue")
    print('Start inference on dataset to get evaluation.')
    predictions = inference_on_dataset(trainer.model, inference_loader, evaluator, overwrite=True)
print("All done.")
