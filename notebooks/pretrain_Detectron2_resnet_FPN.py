#!/usr/bin/env python
# coding: utf-8

# # install timm and lightly pip package
# !pip install -q lightly
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Download detectron yaml config files
# import os
# os.makedirs('/content/COCO-Detection',exist_ok=True)
# !wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/Base-RCNN-FPN.yaml
# !wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
# !mv faster_rcnn_R_50_FPN_1x.yaml COCO-Detection/


# # Imports


################################## Set to True to go through pretraining
################################## Set to False to load previous pretraining attempt
train = False
##################################

import torch
import torch.nn as nn
import torchvision
import lightly

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.modeling import build_model

# Import the Python frameworks we need
import argparse
import numpy as np
import os
import glob
import time
import pytorch_lightning as pl
#import matplotlib.pyplot as plt
#import matplotlib.offsetbox as osb
#from matplotlib import rcParams as rcp
#from sklearn.neighbors import NearestNeighbors
#from sklearn.preprocessing import normalize
#from sklearn import random_projection
#from PIL import Image
from copy import deepcopy
# for resizing images to thumbnails
#import torchvision.transforms.functional as functional


parser = argparse.ArgumentParser(description="""Pretrain resnet50""")
parser.add_argument('-n','--notrain', help='This flag prevents training and loads a previous trained model',
        dest='train', action='store_false', default=True)
parser.add_argument('-g','--gpus', help='Specify max number of gpus', default='1')
parser.add_argument('-e','--epochs', help='Specify max number of epochs', default='10')
parser.add_argument('-b','--batchsize', help='Specify batchsize', default='16')
parser.add_argument('-w','--workers', help='Specify max number of workers', default='10')
parser.add_argument('-p','--basepath', help='Specify base path', default='/data/mostertrij')
args = vars(parser.parse_args())
train = bool(args['train'])
base_path = args['basepath']
# # Load our Detectron2 backbone (bottom-up)

# Configuration
# The default configuration with a batch size of 256 and input resolution of 128
# requires 6GB of GPU memory.
num_workers = int(args['workers'])
batch_size = int(args['batchsize'])
seed = 42
max_epochs = int(args['epochs'])
input_size = 256
gpus = int(args["gpus"]) if torch.cuda.is_available() else 0
print(f"Training is set to:", train)
print(f"Using {num_workers} CPU-workers, {gpus} GPUs, {batch_size} batch size, for {max_epochs} epochs.")


cfg = get_cfg()
conf = '/content/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
conf = '../configs/lofar_detection/uLB300_faster_R50_removed_constantLR.yaml'
cfg.merge_from_file(conf)
detmodel = build_model(cfg) #'detectron2://ImageNetPretrained/MSRA/R-50.pkl'

# Model contains a backbone: detmodel.backbone
# a RPN (region proposal network): detmodel.proposal_generator
# and a ROI (region of interest): detmodel.head roi_heads
# The backbone itself is a resnet-FPN and I assume it is easiest to 
# train the bottom-up part of the resnet-FPN, as that is basically a resnet.
print(detmodel.backbone.bottom_up)
# Create backbone similar to "Lightly Backbone Playground.ipynb" colab
backbone = nn.Sequential(
    # Shape after bottom-up is (num_images,2048,8,8)
    *list(detmodel.backbone.bottom_up.children()),
    # Shape after average pool is (num_images,2048)
    nn.AdaptiveAvgPool2d(1), # output is [bsz, 2048]
)


# # Build SimCLR model on top

# In[4]:


# create our lightly model
# since we use a plain resnet50 the output dimensionality is 2048
# we need to make sure to pass this number of features to lightly
simCLR = lightly.models.SimCLR(backbone, num_ftrs=2048)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device = "cpu" # Helps in debugging
simCLR.to(device)

# let's test our model with random input similar to "Lightly Backbone Playground.ipynb" colab
a = 256
number_of_input_images = 11
number_of_image_channels = 3
toy_input_view_a = torch.randn((number_of_input_images,number_of_image_channels,a,a)).to(device)
toy_input_view_b = torch.randn((number_of_input_images,number_of_image_channels,a,a)).to(device)
out = simCLR(toy_input_view_a, toy_input_view_b)


# # Prep LoTSS data for training

# In[ ]:





# # Train SimCLR model with toy dataset

# In[5]:


# %%
# Let's set the seed for our experiments
pl.seed_everything(seed)

# Make sure `path_to_data` points to the downloaded clothing dataset.
# You can download it using 
# `git clone https://github.com/alexeygrigorev/clothing-dataset.git`
path_to_data = os.path.join(base_path, 'data/frcnn_images/uLB300_removed_noRot/LGZ_COCOstyle/all')


# Setup data augmentations and loaders
# ------------------------------------
#
# The images from the dataset have been taken from above when the clothing was 
# on a table, bed or floor. Therefore, we can make use of additional augmentations
# such as vertical flip or random rotation (90 degrees). 
# By adding these augmentations we learn our model invariance regarding the 
# orientation of the clothing piece. E.g. we don't care if a shirt is upside down
# but more about the strcture which make it a shirt.
# 
# You can learn more about the different augmentations and learned invariances
# here: :ref:`lightly-advanced`.
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    cj_prob = 0.2, # color jitter
    cj_strength=0.1, 
    min_scale=0.6, # minimum scale of random crops
    random_gray_scale = 0,
    gaussian_blur = 0.3,
    kernel_size = 0.1,
    hf_prob=0.5, # horizontal flip
    vf_prob=0.5, # vertical flip
    rr_prob=0.5 # random 90deg rotate
)

# We create a torchvision transformation for embedding the dataset after 
# training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# %%
# We now use the SelfSupervisedEmbedding class from the embedding module.
# First, we create a criterion and an optimizer and then pass them together
# with the model and the dataloader.
criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(simCLR.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    simCLR,
    criterion,
    optimizer,
    dataloader_train_simclr
)

# %% 
# use a GPU if available
print(f"Trying to use {gpus} gpus.")
# %%
# Train the Embedding
# --------------------
# The encoder itself wraps a PyTorch-Lightning module. We can pass any 
# lightning trainer parameter (e.g. gpus=, max_epochs=) to the train_embedding method.
pretrained_model_save_path = os.path.join(base_path,
        f'data/pretrained_models/pretrained_model_epochs{max_epochs}_batch{batch_size}.pth')
if train:
    start = time.time()
    encoder.train_embedding(gpus=gpus, 
                            progress_bar_refresh_rate=100,
                            max_epochs=max_epochs)
    print(f"Training for {max_epochs} epochs took {time.time()-start:.1f} sec.")
    print(f"Or {(time.time()-start)/max_epochs:.1f} sec per epoch.")

    # Storing the simCLR pretrained backbone without the added averagepool-layer
    state_dict = {'resnet50_parameters': simCLR.backbone[:-1].state_dict()}
    torch.save(state_dict, pretrained_model_save_path)
else:
    # Initialize a detectron2 model
    new_detmodel = build_model(cfg)
    # Load the pretrained model
    ckpt = torch.load(pretrained_model_save_path)
    state = ckpt['resnet50_parameters']

    # Change keynames of simCLR pretrained model state_dict to match detectron2 state_dict
    # Sidenote: Iterating over state is equivalent to iterating over state.keys()
    pretrained_keys = deepcopy(list(state.keys()))
    destination_keys = deepcopy(list(new_detmodel.backbone.bottom_up.state_dict()))
    for i, (old_key, dest_key) in enumerate(zip(pretrained_keys, destination_keys)): 
        if i<3: print(old_key,' ,  ',dest_key) # Show the slight difference in keynames
        assert old_key.split('.')[1:] == dest_key.split('.')[1:]
        state[dest_key] = state[old_key]
    [state.pop(k) for k in pretrained_keys]; # Delete old keys

    # Load partial model weights
    new_detmodel.backbone.bottom_up.load_state_dict(state,strict=False)
    
    backbone = nn.Sequential(
        # Shape after bottom-up is (num_images,2048,8,8)
        *list(new_detmodel.backbone.bottom_up.children()),
        # Shape after average pool is (num_images,2048)
        nn.AdaptiveAvgPool2d(1), # output is [bsz, 2048]
    )
    # create our lightly model
    # since we use a plain resnet50 the output dimensionality is 2048
    # we need to make sure to pass this number of features to lightly
    simCLR = lightly.models.SimCLR(backbone, num_ftrs=2048)


# # Load pre-trained weights into Detectron2 model
"""

######### Usually in a new file ####################
# Initialize a detectron2 model
new_detmodel = build_model(cfg)
# Load the pretrained model
ckpt = torch.load('pretrained_model.pth')
state = ckpt['resnet50_parameters']

# Change keynames of simCLR pretrained model state_dict to match detectron2 state_dict
# Sidenote: Iterating over state is equivalent to iterating over state.keys()
pretrained_keys = deepcopy(list(state.keys()))
destination_keys = deepcopy(list(new_detmodel.backbone.bottom_up.state_dict()))
for i, (old_key, dest_key) in enumerate(zip(pretrained_keys, destination_keys)): 
    if i<3: print(old_key,' ,  ',dest_key) # Show the slight difference in keynames
    assert old_key.split('.')[1:] == dest_key.split('.')[1:]
    state[dest_key] = state[old_key]
[state.pop(k) for k in pretrained_keys]; # Delete old keys

# Load partial model weights
new_detmodel.backbone.bottom_up.load_state_dict(state,strict=False)
"""
