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

assert len(argv) > 1, "Insert path of configuration file when executing this script"
cfg = get_cfg()
cfg.merge_from_file(argv[1])
lotss_dr2_path = '/data/mostertrij/data/catalogues/LoTSS_DR2_v100.srl.h5'
if len(argv) == 3:
    start_dir = argv[2]
    print("Beginning of paths:", start_dir)
    cfg.DATASET_PATH = cfg.DATASET_PATH.replace("/data/mostertrij",start_dir)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/data/mostertrij",start_dir)
    cfg.DATASETS.IMAGE_DIR = cfg.DATASETS.IMAGE_DIR.replace("/data/mostertrij",start_dir)
    lotss_dr2_path = lotss_dr2_path.replace("/data/mostertrij",start_dir)
assert os.path.exists(lotss_dr2_path), lotss_dr2_path
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

        if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
            if len(argv) == 3:
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data2/mostertrij",start_dir)
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
            new_data.append(dataset_dicts[i])
            counter+=1

    print('len dataset is:', len(new_data), annotation_filepath)
    return new_data

# Register data inside detectron
# With DATASET_SIZES one can limit the size of these datasets
d = "inference"
inference_dict = get_lofar_dicts(os.path.join(DATASET_PATH,"VIA_json_inference.pkl")) 
DatasetCatalog.register(d, lambda d=d: inference_dict)
MetadataCatalog.get(d).set(thing_classes=["radio_source"])
lofar_metadata = MetadataCatalog.get(d)


print("Sample and plot input data as sanity check")
#"""
for i, dic in enumerate(random.sample(inference_dict, 3)):
    print(dic["file_name"])
    img = imread(dic["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=lofar_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(dic)
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
#pretrained_model_path = "/data/mostertrij/tridentnet/output/v4_all_withRot/model_0059999.pth".replace('/data/mostertrij',start_dir)
pretrained_model_path = "/data/mostertrij/tridentnet/output/v3_precomputed_constantLR_withRot_no_box_reg/model_0005999.pth".replace('/data/mostertrij',start_dir)
print("Load model:", pretrained_model_path)
cfg.MODEL.WEIGHTS = os.path.join(pretrained_model_path)  # path to the model we just trained
trainer = LOFARTrainer(cfg) 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer.resume_or_load(resume=True)

print('Load inference loader.')
inference_loader = build_detection_test_loader(cfg, d)
print('Load LOFAR evaluator.')
evaluator = LOFAREvaluator(d, cfg.OUTPUT_DIR, distributed=True, inference_only=True,
        kafka_to_lgm=False,component_save_name="bare_predicted_component_catalogue")
print('Start inference on dataset.')
predictions = inference_on_dataset(trainer.model, inference_loader, evaluator, overwrite=True)
print('Done with inference.')

def get_idx_dict(cat):
    """Create dict that returns row index of objects when given sourcename"""
    idx_dict = {s:idx for s, idx in zip(cat.Source_Name.values,cat.index.values)}
    return idx_dict

def get_comps(predicted_comp_cat):
    """Turn list of source_names, component_names into a list"""
    combined_names = list(OrderedDict.fromkeys(predicted_comp_cat['Source_Name'].values))
    comp_dict = {s:[] for s in combined_names}
    for s, comp in zip(predicted_comp_cat.Source_Name.values,predicted_comp_cat.Component_Name.values):
            comp_dict[s].append(comp)
    return [comp_dict[n] for n in combined_names]

def ellipse(x0,y0,a,b,pa,n=200):
    theta=np.linspace(0,2*np.pi,n,endpoint=False)
    st = np.sin(theta)
    ct = np.cos(theta)
    pa = np.deg2rad(pa+90)
    sa = np.sin(pa)
    ca = np.cos(pa)
    p = np.empty((n, 2))
    p[:, 0] = x0 + a * ca * ct - b * sa * st
    p[:, 1] = y0 + a * sa * ct + b * ca * st
    return Polygon(p)

class Make_Shape(object):
    '''Basic idea taken from remove_lgz_sources.py -- maybe should be merged with this one day
    but the FITS keywords are different.
    '''
    def __init__(self,clist):
        '''
        clist: a list of components that form part of the source, with RA, DEC, DC_Maj...
        '''
        ra=np.mean(clist['RA'])
        dec=np.mean(clist['DEC'])

        ellist=[]

        for ir, r in clist.iterrows():
            n_ra=r['RA']
            n_dec=r['DEC']
            x=3600*np.cos(dec*np.pi/180.0)*(ra-n_ra)
            y=3600*(n_dec-dec)
            newp=ellipse(x,y,r['DC_Maj']+0.1,r['DC_Min']+0.1,r['PA'])
            ellist.append(newp)
        self.cp=cascaded_union(ellist)
        self.ra=ra
        self.dec=dec
        self.h=self.cp.convex_hull
        a=np.asarray(self.h.exterior.coords)
        #for i,e in enumerate(ellist):
        #    if i==0:
        #        a=np.asarray(e.exterior.coords)
        #    else:
        #        a=np.append(a,e.exterior.coords,axis=0)
        mdist2=0
        bestcoords=None
        for r in a:
            dist2=(a[:,0]-r[0])**2.0+(a[:,1]-r[1])**2.0
            idist=np.argmax(dist2)
            mdist=dist2[idist]
            if mdist>mdist2:
                mdist2=mdist
                bestcoords=(r,a[idist])
        self.mdist2=mdist2
        self.bestcoords=bestcoords
        self.a=a

    def length(self):
        return np.sqrt(self.mdist2)

    def pa(self):
        p1,p2=self.bestcoords
        dp=p2-p1
        angle=(180*np.arctan2(dp[1],dp[0])/np.pi)-90
        if angle<-180:
            angle+=360
        if angle<0:
            angle+=180
        return angle

    def width(self):
        p1,p2=self.bestcoords
        d = np.cross(p2-p1, self.a-p1)/self.length()
        return 2*np.max(d)
        
def sourcename(ra,dec):
    sc=SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
    s=sc.to_string(style='hmsdms',sep='',precision=2)
    return str('ILTJ'+s).replace(' ','')[:-1]

def save_cat(dataframe, output_dir, save_name, keys=None):
    # Save to hdf
    hdf_path = os.path.join(output_dir, save_name + '.h5')
    dataframe.to_hdf(hdf_path,'df')
    # Save to fits
    fits_path = os.path.join(output_dir, save_name + '.fits')
    if os.path.exists(fits_path):
        print("Fits file exists. Overwriting it now")
        # Remove old fits file
        os.remove(fits_path)
    t = Table.from_pandas(dataframe)
    t.write(fits_path, format='fits')

comp_keys = ['Source_Name', 'RA', 'E_RA', 'DEC', 'E_DEC', 'Peak_flux',
       'E_Peak_flux', 'Total_flux', 'E_Total_flux', 'Maj', 'E_Maj', 'Min',
       'E_Min', 'DC_Maj', 'E_DC_Maj', 'DC_Min', 'E_DC_Min', 'PA', 'E_PA',
       'DC_PA', 'E_DC_PA', 'Isl_rms', 'S_Code', 'Mosaic_ID',
       'Component_Name']

# Read cats
lotss_dr2_cat = pd.read_hdf(lotss_dr2_path)
predicted_comp_cat = pd.read_hdf(os.path.join(cfg.OUTPUT_DIR, 'bare_predicted_component_catalogue.h5'))

# Get unique combined names
comp_names = get_comps(predicted_comp_cat)

# Get component lists for each predicted source
idx_dict = get_idx_dict(lotss_dr2_cat)
clists = [lotss_dr2_cat.iloc[[idx_dict[c] for c in comps]] for comps in comp_names]


# Iterate over clists
# Modelled after https://github.com/mhardcastle/lotss-catalogue/blob/master/catalogue_create/process_lgz.py
# Got it working for everything but size

keys = ['Source_Name','RA','E_RA','DEC','E_DEC','Peak_flux','E_Peak_flux','Total_flux','E_Total_flux',
        'Isl_rms','S_Code','Mosaic_ID']
#predicted_source_cat = pd.DataFrame(columns=keys, data={})
data = []
compdata = []
for irow, clist in enumerate(clists):
    #clist = clist_series.to_dict('records')

    ms=Make_Shape(clist)
    r = {}
    r['Predicted_Size']=ms.length()
    r['Predicted_Width']=ms.width()
    r['Predicted_PA']=ms.pa()
    # WHETHER THE SOURCE HAS CHANGED OR NOT,
    # recreate all of the source's radio info from its component list
    # this ensures consistency

    r['Predicted_Assoc']=len(clist)
    if len(clist)==1:
        # only one component, so we can use its properties
        c=clist.iloc[0]
        for key in keys:
            r[key]=c[key]
#         r['Size']=c['DC_Maj']
#         if size is not None:
#             r['Size']=size
#             if size>r['Predicted_Size']:
#                 r['Predicted_Size']=size
        rc = deepcopy(c)
        sname = sourcename(rc.RA, rc.DEC)
        rc['Component_Name'] = rc['Source_Name']
        rc['Source_Name'] = sname
        compdata.append(rc)
    else:
        tfluxsum=clist['Total_flux'].sum()
        ra=np.sum(clist['RA']*clist['Total_flux'])/tfluxsum
        dec=np.sum(clist['DEC']*clist['Total_flux'])/tfluxsum
        sname=sourcename(ra,dec)
        for icomp, comp in clist.iterrows():
            rc = deepcopy(comp)
            rc['Component_Name'] = rc['Source_Name']
            rc['Source_Name'] = sname
            compdata.append(rc)
        r['RA']=ra
        r['DEC']=dec
        r['Source_Name']=sname
        r['E_RA']=np.sqrt(np.mean(np.power(clist['E_RA'],2)))
        r['E_DEC']=np.sqrt(np.mean(np.power(clist['E_DEC'],2)))
        r['Source_Name']=sname
        r['Total_flux']=tfluxsum
        r['E_Total_flux']=np.sqrt(np.sum(np.power(clist['E_Total_flux'],2)))
        maxpk=clist['Peak_flux'].idxmax()
        r['Peak_flux']=clist.loc[maxpk]['Peak_flux']
        r['E_Peak_flux']=clist.loc[maxpk]['E_Peak_flux']
        r['S_Code']='M'
        r['Isl_rms']=np.mean(clist['Isl_rms'])
        r['Mosaic_ID']=clist.loc[maxpk]['Mosaic_ID']
#         seps=[]
#         for c in clist:
#             seps.append(separation(c['RA'],c['DEC'],clist['RA'],clist['DEC']))
#         maxsep=np.max(seps)*scale
#         maxsize=np.max(clist['Maj'])
#         maxsize=max((maxsep,maxsize))
#         if size is not None:
#             if size>maxsize:
#                 maxsize=size
#             if size>r['New_size']:
#                 r['New_size']=size

#         print '      sizes:',maxsep,maxsize
#         r['Size']=maxsize
    #rdf = pd.DataFrame(data=r, index=[0])
    #print(rdf[keys])
    #predicted_source_cat.append(rdf[keys], ignore_index=True)
    data.append(r)
source_component_cat = pd.DataFrame(columns=comp_keys, data=compdata).reset_index(drop=True)
predicted_source_cat = pd.DataFrame(columns=keys+['Predicted_Size','Predicted_Width','Predicted_PA',
                                                  'Predicted_Assoc'], data=data)
save_cat(predicted_source_cat, cfg.OUTPUT_DIR, 'LoTSS_predicted_v0_merge',
         keys=keys+['Predicted_Size','Predicted_Width','Predicted_PA','Predicted_Assoc'] )
save_cat(source_component_cat, cfg.OUTPUT_DIR, 'LoTSS_predicted_v0.comp', keys=comp_keys)

print("Inference catalogues created and saved at:", cfg.OUTPUT_DIR)
print("All done.")
