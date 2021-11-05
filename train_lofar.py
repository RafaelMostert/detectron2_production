#!/usr/bin/env python

# # Import and load Detectron2 and libraries
#import torch, torchvision


print("Setup detectron2 logger")
from detectron2.utils.logger import setup_logger
setup_logger()
print("Import some common detectron2 utilities")
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, LOFARTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# import some common libraries
import numpy as np
import time
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
print(cfg.TEST.REMOVE_THRESHOLD,type(cfg.TEST.REMOVE_THRESHOLD))
cfg.merge_from_file(argv[1])
print("after readfromfile:", cfg.TEST.REMOVE_THRESHOLD,type(cfg.TEST.REMOVE_THRESHOLD))
if len(argv) >= 3:
    start_dir = argv[2]
    print("Beginning of paths:", start_dir)
    cfg.DATASET_PATH = cfg.DATASET_PATH.replace("/data/mostertrij",start_dir)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/data/mostertrij",start_dir)
    cfg.MODEL.PRETRAINED_WEIGHTS = cfg.MODEL.PRETRAINED_WEIGHTS.replace("/data/mostertrij",start_dir)
    cfg.DATASETS.IMAGE_DIR = cfg.DATASETS.IMAGE_DIR.replace("/data/mostertrij",start_dir)
    if len(argv) >= 4:
        cfg.SEED = int(argv[3])
        if cfg.OUTPUT_DIR.endswith('/'):
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR[:-1] + f'_seed{cfg.SEED}'
        else:
            cfg.OUTPUT_DIR += f'_seed{cfg.SEED}'
    print("Training seed is:", cfg.SEED)
print(f"Loaded configuration file {argv[1]}")
#ROTATION_ENABLED = bool(int(argv[2])) # 0 is False, 1 is True
DATASET_PATH= cfg.DATASET_PATH
print(f"Experiment: {cfg.EXPERIMENT_NAME}")
if not cfg.MODEL.PRETRAINED_WEIGHTS == "":
    print(f"Pretrained weights loaded from: {cfg.MODEL.PRETRAINED_WEIGHTS}")

print(f"Rotation enabled: {cfg.INPUT.ROTATION_ENABLED}")
print(f"Precomputed bboxes: {cfg.MODEL.PROPOSAL_GENERATOR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
if len(argv) == 5:
    train_dataset_size = int(argv[4])
    cfg.DATASETS.TRAIN_SIZE = train_dataset_size
    print(f"Training data set contains {cfg.DATASETS.TRAIN_SIZE} images.")
    if cfg.OUTPUT_DIR.endswith('/'):
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR[:-1] + f'_{train_dataset_size}' 
    else:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f'_{train_dataset_size}'
print(f"Output path: {cfg.OUTPUT_DIR}")
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)


print("Load our data")
#def get_lofar_dicts(annotation_filepath, n_cutouts=np.inf, rotation=False):
def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    new_data = []
    train_size_limited = False
    if annotation_filepath.endswith('train.pkl'): 
        if cfg.DATASETS.TRAIN_SIZE < len(dataset_dicts) and cfg.INPUT.ROTATION_ENABLED:
            assert cfg.INPUT.ROTATION_ENABLED, ("limiting the size of the training set is only", \
                "implemented with rotation augmentation enabled.")
            # for a filename like /data/ILTJ130552.61+495745.5_radio_DR2_rotated0deg.png
            # this leaves us with a list of names like ILTJ130552.61+495745.5 
            component_names = list([dataset_dicts[i]['file_name'].split('/')[-1].split('_')[0]
                    for i in range(len(dataset_dicts)) 
                    if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png')])
            assert len(component_names) == len(set(component_names)), "duplicate sources entering training set."
            component_names = component_names[:cfg.DATASETS.TRAIN_SIZE]
            print("Debug, first ten component names:",component_names[:10])
            train_size_limited = True
            

    for i in range(len(dataset_dicts)):
        component_name = dataset_dicts[i]['file_name'].split('/')[-1].split('_')[0]
        for ob in dataset_dicts[i]['annotations']:
            ob['bbox_mode'] = BoxMode.XYXY_ABS
        if cfg.MODEL.PROPOSAL_GENERATOR:
            dataset_dicts[i]["proposal_bbox_mode"] = BoxMode.XYXY_ABS
        if cfg.INPUT.ROTATION_ENABLED:
            if len(argv) >= 3:
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace('/data2/','/data/').replace("/data/mostertrij",start_dir)
            if not train_size_limited or component_name in component_names:
                new_data.append(dataset_dicts[i])
        else:
            if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
                if len(argv) >= 3:
                    dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace('/data2/','/data/').replace("/data/mostertrij",start_dir)
                new_data.append(dataset_dicts[i])
    print('len dataset is:', len(new_data), annotation_filepath)
    return new_data

# Register data inside detectron
# With cfg.DATASETS.TRAIN_SIZE one can limit the size of the train dataset
for d in ["train", "val", "test"]:
    DatasetCatalog.register(d, 
                            lambda d=d:
                            get_lofar_dicts(os.path.join(
                                DATASET_PATH,f"VIA_json_{d}.pkl")))
    MetadataCatalog.get(d).set(thing_classes=["radio_source"])
lofar_metadata = MetadataCatalog.get("train")


print("Sample and plot input data as sanity check")
#"""
train_dict = get_lofar_dicts(os.path.join(DATASET_PATH,"VIA_json_train.pkl")) 
for i, d in enumerate(random.sample(train_dict, 3)):
    #for i, d in enumerate(train_dict):
    img = imread(d["file_name"])
    print('img filename:', d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=lofar_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    a= vis.get_image()[:, :, ::-1]
    plt.figure(figsize=(10,10))
    plt.imshow(a)
    # overplot the best bbox
    box = d['annotations'][0]['bbox']
    #if d["file_name"].endswith('ILTJ110530.36+465055.8_radio_DR2_rotated0deg.png'):
    #    print("Proposal boxes:")
    #    print(d['proposal_boxes'])
    box = d['annotations'][0]['bbox']
    color='k'
    plt.plot([box[0],box[0]],[box[1],box[3]],color=color)
    plt.plot([box[2],box[2]],[box[1],box[3]],color=color)
    plt.plot([box[0],box[2]],[box[1],box[1]],color=color)
    plt.plot([box[0],box[2]],[box[3],box[3]],color=color)

    plt.savefig(os.path.join(cfg.OUTPUT_DIR,f"sanity_check_{i}_{d['file_name'].split('/')[-1].replace('.png','')}.jpg"))
    plt.close()
#"""

# # Train mode

# To implement the LOFAR relevant metrics I changed
# DefaultTrainer into LOFARTrainer
# where the latter calls LOFAREvaluator within build_hooks instead of the default evaluator
# this works for the after the fact test eval
# for train eval those things are somewhere within a model 
# specifically a model that takes data and retuns a dict of losses
print("Load model")
trainer = LOFARTrainer(cfg) 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
pretrained_model_path = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
if os.path.exists(pretrained_model_path):
    cfg.MODEL.WEIGHTS = pretrained_model_path
    trainer.resume_or_load(resume=True)
else:
    trainer.resume_or_load(resume=False)

print("Start training")
start = time.time()
trainer.train()
print(f"Training took {(time.time()-start)/3600:.2f} hour")
print('Done training.')

"""
# Look at training curves in tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
#%tensorboard --logdir output --host "0.0.0.0" --port 6006
get_ipython().run_line_magic('tensorboard', '--logdir output  --port 6006')
# In local command line input 
#ssh -X -N -f -L localhost:8890:localhost:6006 tritanium
# Then open localhost:8890 to see tensorboard
"""

