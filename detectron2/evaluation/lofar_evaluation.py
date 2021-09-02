# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from operator import itemgetter
import contextlib
import pickle
import pandas as pd
import io
import numpy as np
import torch
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict, Iterable
from fvcore.common.file_io import PathManager
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class LOFAREvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, 
    outputs using LOFAR relevant metrics.
    The relevant metric measures whether a proposed detection box for the central source is able to
    capture all and only the sources associated to a single source as determined by crowdsourced
    associations in LGZ.
    That is: for all proposed boxes that cover the middle pixel of the input image check which
    sources from the component catalogue are inside. 
    The predicted box can fail in three different ways:
    1. No predicted box covers the middle pixel
    2. The predicted box misses a number of components
    3. The predicted box encompasses too many components
    4. The prediction score for the predicted box is lower than other boxes that cover the middle
        pixel
    5. The prediction score is lower than x
    """

    def __init__(self, dataset_name, output_dir, distributed=True, inference_only=False,
            remove_unresolved=False, sigmabox=5,
            segmentation_dir='/data1/mostertrij/data/cache/segmentation_maps',
            kafka_to_lgm=False,component_save_name=None,debug=False, save_predictions=False,
            cats='spring'):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        #self._metadata = MetadataCatalog.get(dataset_name)
        """
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        """
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._predictions_json = os.path.join(output_dir, "predictions.json")
        self.inference_only=inference_only
        self.kafka_to_lgm = kafka_to_lgm
        self.save_name = component_save_name
        self.save_predictions = save_predictions
        self.remove_unresolved = remove_unresolved
        self.segmentation_dir = segmentation_dir
        self.debug = debug
        self.related_unresolved = []
        self.unrelated_unresolved = []
        self.wide_focus = []
        self.old_related_unresolved = []
        self.old_unrelated_unresolved = []
        self.cats = cats
        self.sigmabox = sigmabox

    def reset(self):
        self._predictions = []
        self.focussed_comps = []
        self.related_comps = []
        self.unrelated_comps = []
        self.unrelated_names = []
        self.focussed_names = []
        self.n_comps = []
        self.pred_bboxes_scores = []

        self.related_unresolved = []
        self.unrelated_unresolved = []
        self.wide_focus = []
        self.old_related_unresolved = []
        self.old_unrelated_unresolved = []

        # -1 unknown, 0  good association, 1 single missing, 2 single
        # overestimated, 3 multi underestimated, 4 multi overestimated
        self.misboxed_category = [] 

    def process(self, inputs, outputs):
        # Save ground truths and predicted bounding boxes to this class
        for input, output in zip(inputs, outputs):
            if self.inference_only:
                prediction = {"image_id": input["image_id"], "file_name":input["file_name"],
                    "focussed_comp":input["focussed_comp"],"related_comp":input["related_comp"],
                    "unrelated_comp":input["unrelated_comp"],
                    "related_unresolved":input["related_unresolved"],
                    "unrelated_unresolved":input["unrelated_unresolved"],
                    "wide_focus":input["wide_focus"],
                    "unrelated_names":input["unrelated_names"]}
            else:
                prediction = {"image_id": input["image_id"], "file_name":input["file_name"],
                    "focussed_comp":input["focussed_comp"],"related_comp":input["related_comp"],
                    "unrelated_comp":input["unrelated_comp"],
                    "related_unresolved":input["related_unresolved"],
                    "unrelated_unresolved":input["unrelated_unresolved"],
                    "wide_focus":input["wide_focus"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)#.numpy()
                prediction["instances"] = instances
            self._predictions.append(prediction)

        self.focussed_names = [p["file_name"].split('/')[-1].split('_')[0]
                for p in self._predictions]
        #print("prediction_shape", np.shape(self._predictions), self._predictions[0])
        self.focussed_comps = [p["focussed_comp"] for p in self._predictions]
        self.related_comps = [p["related_comp"] for p in self._predictions]
        self.unrelated_comps = [p["unrelated_comp"] for p in self._predictions]
        if self.inference_only:
            self.misboxed_category = np.array([-1 for p in self._predictions])
        else:
            self.misboxed_category = np.array([0 for p in self._predictions])
        if self.remove_unresolved:
            self.related_unresolved = [p["related_unresolved"] for p in self._predictions]
            self.unrelated_unresolved = [p["unrelated_unresolved"] for p in self._predictions]
            self.wide_focus = [p["wide_focus"] for p in self._predictions]
            self.old_related_unresolved = copy.deepcopy(self.related_unresolved)
            self.old_unrelated_unresolved = copy.deepcopy(self.unrelated_unresolved)
            #print("related unresolved:", self.related_unresolved[0])
            #print("unrelated unresolved:", self.unrelated_unresolved[0])

        if self.inference_only:
            self.unrelated_names = [p["unrelated_names"] if len(p["unrelated_names"])>0 else [] 
                for p in self._predictions]
        self.n_comps = [1+len(c[0]) if len(c[0])>0 else 1 for c in self.related_comps]
        
        # Get predicted bounding boxes per image as numpy arrays
        self.pred_bboxes_scores = [(image_dict['instances'].get_fields()['pred_boxes'].tensor.numpy(), 
                  image_dict['instances'].get_fields()['scores'].numpy()) 
                 for image_dict in self._predictions]
        for image_dict in self._predictions:
            box = image_dict['instances'].get_fields()['pred_boxes'].tensor.numpy()
            score = image_dict['instances'].get_fields()['scores'].numpy()
            #if len(box) == 0:
            #    print("box, score:", box, score)
            #    print("image_dict:", image_dict['instances'])
            #    sdfsdf

    def evaluate(self):
        # for parallel execution 
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            #self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        # Return empty if inputs and outputs are non-existing
        if len(self._predictions) == 0:
            logger.warning("[LOFAREvaluator] Did not receive valid predictions.")
            return {}

        # Save predicted instances
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)



        if self.inference_only:
            return copy.deepcopy(self.return_component_list())
        else:
            includes_associated_fail_fraction, includes_unassociated_fail_fraction = \
                self._evaluate_predictions_on_lofar_score()

            # Calculate/print catalogue improvement
            base_score = self.baseline()
            correct_cat = self.our_score(includes_associated_fail_fraction, includes_unassociated_fail_fraction)
            self.improv(base_score, correct_cat)

            self._results = OrderedDict()
            self._results["bbox"] = {"assoc_single_fail_fraction": includes_associated_fail_fraction[0],
            "assoc_multi_fail_fraction": includes_associated_fail_fraction[1],
            "unassoc_single_fail_fraction": includes_unassociated_fail_fraction[0],
            "unassoc_multi_fail_fraction": includes_unassociated_fail_fraction[1],
            "correct_catalogue": correct_cat}
            # Copy so the caller can do whatever with results
            return copy.deepcopy(self._results)

    def baseline(self):
        total = len(self.n_comps) #self.single_comps + self.multi_comps
        correct = self.single_comps/total
        print(f"Baseline assumption cat is {correct:.1%} correct")
        return correct

    def our_score(self,assoc_fail, unassoc_fail, suffix=''):
        fail_single = assoc_fail[0]*self.single_comps + unassoc_fail[0]*self.single_comps
        fail_multi = assoc_fail[1]*self.multi_comps + unassoc_fail[1]*self.multi_comps
        total = len(self.n_comps) # self.single_comps + self.multi_comps
        correct = (total-(fail_single+fail_multi))/total
        print(f"{self._dataset_name} cat is {correct:.1%} correct")
        return correct

    def improv(self, baseline, our_score):
        print(f"{(our_score-baseline)/baseline:.2%} improvement")

    def area(self, bbox):
        return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])

    def flatten_yield(self, l):
        '''Flatten a list or numpy array'''
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, 
                    (str, bytes)):
                yield from self.flatten_yield(el)
            else:
                yield el     

    def is_within(self, x,y,xmin,ymin,xmax,ymax):
        """Return true if x, y lies within xmin,ymin,xmax,ymax.
        False otherwise.
        """
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
        else:
            return False


    def return_component_list(self, scale_factor=1, debug=False, imsize=200):
        """ 
        return component list in case of inference
        """
        print("Return component list")

        # Filter out predicted bboxes that do not cover the focussed pixel
        pred_central_bboxes_scores = [[(tuple(bbox),score) for bbox, score in zip(bboxes, scores) 
                            if self.is_within(x*scale_factor,y*scale_factor, 
                                bbox[0],bbox[1],bbox[2],bbox[3])] 
                              for (x, y), (bboxes, scores) 
                              in zip(self.focussed_comps, self.pred_bboxes_scores)]
        
        # Record for which images we have no central bbox
        self.central_covered = [True if len(bboxes_scores) > 0 else False 
                                      for bboxes_scores in pred_central_bboxes_scores]
        
        # Take only the highest scoring bbox from this list of bboxes
        max_score_threshold=0.05
        self.second_best = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[1:4] 
                                      if len(bboxes_scores) > 1 else [[[-1,-1,-1,-1],-99]] 
                                      for bboxes_scores in pred_central_bboxes_scores]
        self.pred_central_bboxes_scores = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[0] 
                                      if len(bboxes_scores) > 0 else [[-1,-1,-1,-1],0] 
                                      for bboxes_scores in pred_central_bboxes_scores]
        #max_scores =[score for bbox,score in self.pred_central_bboxes_scores] 
        #self.second_best = [[(box,score) for box,score in bboxes_scores if score >
        #    max_score-max_score_threshold]
        #         for bboxes_scores,max_score in zip(self.second_best,max_scores)]

        # Check which components fall within the predicted bbox with the highest score
        if self.remove_unresolved:
            self.reinsert_unresolved_for_triplets()
            self.comp_inside_box = [[foc_name]+[name for x,y,unresolved,name in zip(
                    xs,ys,unresolved_list,names) 
                    if self.is_within(x*scale_factor,y*scale_factor,bbox[0],bbox[1],bbox[2],bbox[3]) \
                                        and not unresolved]
                                for (xs,ys),unresolved_list, names, (bbox, score), foc_name in zip(self.unrelated_comps,
                                    self.unrelated_unresolved,
                                        self.unrelated_names,self.pred_central_bboxes_scores,self.focussed_names)]
        else:
            self.comp_inside_box = [[foc_name]+[name for x,y,name in zip(xs,ys,names) 
                                        if self.is_within(x*scale_factor,y*scale_factor,bbox[0],bbox[1],bbox[2],bbox[3])]
                                for (xs,ys),names, (bbox, score), foc_name in zip(self.unrelated_comps,
                                        self.unrelated_names,self.pred_central_bboxes_scores,self.focussed_names)]

        # Get bbox sizes
        sizes = [self.area(bbox) for bbox, score in self.pred_central_bboxes_scores]
        # Get indices of appearing focussed comp with largest box
        indices = [max([(i,size) for i, (names,size) in enumerate(zip(self.comp_inside_box, sizes)) if
            foc_name in names], key = lambda t: t[1])[0]
            for foc_name in self.focussed_names]
        # remove duplicates
        #indices = list(set(indices))
        indices = list(OrderedDict.fromkeys(indices))
        self.comp_inside_box = np.array(self.comp_inside_box)[indices]
        # Create pandas dataframe 
        combined_names = []
        comp_names = []
        for i, names in enumerate(self.comp_inside_box):
            for name in names:
                combined_names.append(f"Combined_{i}")
                comp_names.append(name)
        comp_df = pd.DataFrame({"Source_Name":combined_names,"Component_Name":comp_names})
        # Save to hdf
        hdf_path = os.path.join(self._output_dir,self.save_name + ".h5")
        comp_df.to_hdf(hdf_path,'df')
        # Save to fits
        fits_path = os.path.join(self._output_dir,self.save_name + ".fits")
        if os.path.exists(fits_path):
            print("Component fits already exists. Overwriting it now")
            # Remove old fits file
            os.remove(fits_path)
        t = Table([combined_names,comp_names], names=('Source_Name', 'Component_Name'))
        t.write(fits_path, format='fits')

        print("Plot first 1000 predictions (predictions not aggregated yet)")
        self.plot_predictions("all_prediction_debug_images",cutout_list=list(range(len(self.related_comps))),
                debug=False, show_second_best=False)
        print("Plot first 1000 final predictions: components should only appear in a single bounding box")
        self.plot_predictions("final_prediction_debug_images",cutout_list=np.array(indices)[:1000], debug=False)
        
        return comp_df


    def reinsert_unresolved_for_triplets(self, imsize=200, debug=False):
        """Given a list of unresolved objects, tries to reinsert them if sensible"""
        # Check if any unresolved components are inside predicted bbox.
        unresolved_in_predicted_bboxes = [False for _ in range(len(self.focussed_names))]
        for i, (bbox, score) in enumerate(self.pred_central_bboxes_scores):
            unresolved_flag = False
            (xs,ys), unresolved_list = self.related_comps[i], self.related_unresolved[i]
            for unresolved, x,y in zip(unresolved_list,xs,ys):
                if self.is_within(x,y,bbox[0],bbox[1],bbox[2],bbox[3]) and unresolved:
                    unresolved_in_predicted_bboxes[i] = True
                    unresolved_flag = True
                    break
            if not unresolved_flag:
                (xs,ys), unresolved_list = self.unrelated_comps[i], self.unrelated_unresolved[i]
                for unresolved, x,y in zip(unresolved_list,xs,ys):
                    if self.is_within(x,y,bbox[0],bbox[1],bbox[2],bbox[3]) and unresolved:
                        unresolved_in_predicted_bboxes[i] = True
                        break

        # Check which resolved components are inside predicted bbox
        any_unresolved_related = [[self.is_within(x,y,
            bbox[0],bbox[1],bbox[2],bbox[3]) and not unresolved \
                    for unresolved, x,y in zip(unresolved_list, xs,ys)] \
                    for (xs,ys), unresolved_list, (bbox,score) \
                    in zip(self.related_comps, self.related_unresolved,
                                    self.pred_central_bboxes_scores)]
        related_resolved_in_predicted_bboxes = []
        for a, (xs,ys) in zip(any_unresolved_related, self.related_comps):
            if np.any(a):
                related_resolved_in_predicted_bboxes.append([xs[a],ys[a]])
            else:
                related_resolved_in_predicted_bboxes.append([[],[]])

        any_unresolved_unrelated = [[self.is_within(x,y,
            bbox[0],bbox[1],bbox[2],bbox[3]) and not unresolved \
                    for unresolved, x,y in zip(unresolved_list, xs,ys)] \
                    for (xs,ys), unresolved_list, (bbox,score) \
                    in zip(self.unrelated_comps, self.unrelated_unresolved,
                                    self.pred_central_bboxes_scores)]
        unrelated_resolved_in_predicted_bboxes = []
        for a, (xs,ys) in zip(any_unresolved_unrelated, self.unrelated_comps):
            if np.any(a):
                unrelated_resolved_in_predicted_bboxes.append([xs[a],ys[a]])
            else:
                unrelated_resolved_in_predicted_bboxes.append([[],[]])

        # If so proceed with following actions.
        segmentation_paths = [os.path.join(self.segmentation_dir, f"{s}_{imsize}_{self.sigmabox}sigma.pkl")
                for s in self.focussed_names]
        new_rel_unresolved_per_cutout, new_unrel_unresolved_per_cutout = [], []
        for t, (action_needed,  related_comp, unrelated_comp) in \
                    enumerate(zip(unresolved_in_predicted_bboxes,
                            related_resolved_in_predicted_bboxes,
                            unrelated_resolved_in_predicted_bboxes)):
            if action_needed:

                # Load segmentation map per cutout
                with open(segmentation_paths[t], 'rb') as f:
                    segmented_cutout = pickle.load(f)

                # Given a segmentation map
                outlines = segmented_cutout.outline_segments()

                # Debug plot outlines
                if debug:
                    plt.imshow(outlines)
                    plt.show()

                # Find segmentation islands corresponding to radio components
                pixel_xs = [self.focussed_comps[t][0]]+self.wide_focus[t][0]+list(related_comp[0])+list(unrelated_comp[0])
                pixel_ys = [self.focussed_comps[t][1]]+self.wide_focus[t][1]+list(related_comp[1])+list(unrelated_comp[1])
                if debug:
                    print("pixel_xs", pixel_xs)
                    print("pixel_ys", pixel_ys)
                try:
                    labels = [segmented_cutout.data[int(round(y)), int(round(x))] for x,y in
                        zip(pixel_xs, pixel_ys)]
                except:
                    if debug:
                        print("no label found")
                    continue
                labels = [l for l in labels if not l == 0]
                labels = np.array(labels)-1 # labels start at 1 as 0 is background

                # If no labels are found, enlarge the search radius
                labels2 = labels
                r = 3 # Search radius
                if labels.size != len(pixel_xs):
                    labels = list(set(self.flatten_yield([
                        segmented_cutout.data[int(round(y))-r:int(round(y))+r,
                            int(round(x))-r:int(round(x))+r] for x,y in \
                        zip(pixel_xs, pixel_ys)])))
                    labels = [l for l in labels if not l == 0]
                    labels = np.array(labels)-1 # labels start at 1 as 0 is background

                if debug:
                    print("labels:", labels)
                # Get bounding boxes for segments
                bboxes = [segmented_cutout.segments[l].bbox for l in labels]

                # Get all non-background points from the segmentation outlines
                points = []
                for b in bboxes:
                    outline = outlines[b.iymin:b.iymax,b.ixmin:b.ixmax]
                    for i, rows in enumerate(outline):
                        for j,el in enumerate(rows):
                            if el > 0:
                                points.append([b.iymin+i,b.ixmin+j])
                points = np.array(points)
                if debug:
                    plt.scatter(*list(zip(*points)))
                    plt.show()

                # Create convex hull from points (removes inner points)
                hull = ConvexHull( points )
                # Create matplotlib Path object from convexhull vertices
                hull_path = Path( points[hull.vertices] )
                # This allows us to query whether a point is contained by our convex hull
                new_rel_unresolved_list = [False for _ in range(len(self.related_unresolved[t]))]
                new_unrel_unresolved_list = [False for _ in range(len(self.unrelated_unresolved[t]))]
                for i, (x,y, unresolved) in enumerate(zip(self.related_comps[t][0],
                    self.related_comps[t][1], self.related_unresolved[t])):
                    if unresolved:
                        should_be_reinstated = hull_path.contains_point((y,x))
                        if debug:
                            print(f"Does our predicted convex hull contain point {x},{y}?")
                            print(should_be_reinstated)
                        if not should_be_reinstated:
                            new_rel_unresolved_list[i] =  True
                for i, (x,y, unresolved) in enumerate(zip(self.unrelated_comps[t][0],
                    self.unrelated_comps[t][1], self.unrelated_unresolved[t])):
                    if unresolved:
                        should_be_reinstated = hull_path.contains_point((y,x))
                        if debug:
                            print(f"Does our predicted convex hull contain point {x},{y}?")
                            print(should_be_reinstated)
                        if not should_be_reinstated:
                            new_unrel_unresolved_list[i] =  True
                new_rel_unresolved_per_cutout.append(new_rel_unresolved_list)
                new_unrel_unresolved_per_cutout.append(new_unrel_unresolved_list)
            else:
                new_rel_unresolved_per_cutout.append(self.related_unresolved[t])
                new_unrel_unresolved_per_cutout.append(self.unrelated_unresolved[t])
        self.related_unresolved = new_rel_unresolved_per_cutout
        self.unrelated_unresolved = new_unrel_unresolved_per_cutout
        

    def _evaluate_predictions_on_lofar_score(self, scale_factor=1, debug=False, imsize=200):
        """ 
        Evaluate the results using our LOFAR appropriate score.

            Evaluate self._predictions on the given tasks.
            Fill self._results with the metrics of the tasks.

            That is: for all proposed boxes that cover the middle pixel of the input image check which
            sources from the component catalogue are inside. 
            The predicted box can fail in three different ways:
            1. No predicted box covers the focussed box
            2. The predicted central box misses a number of components
            3. The predicted central box encompasses too many components
            4. The prediction score for the predicted box is lower than other boxes that cover the middle
                pixel
            5. The prediction score is lower than x
        
        """
        print("Evaluate predictions")
        if debug:
            #Check ground truth and prediction values of first item
            print("scale_factor", scale_factor)
            print("focus, related, unrelated, ncomp")
            print(self.focussed_comps[0], self.related_comps[0], self.unrelated_comps[0], self.n_comps[0])
            #print(np.shape(self.focussed_comps), np.shape(self.related_comps),
            #        np.shape(self.unrelated_comps), np.shape(self.n_comps))
            print("ncomp",self.n_comps)
            #print("pred_bboxes_scores")
            #print(self.pred_bboxes_scores[0])

        
        # Get number of single and multi comp sources
        self.single_comps = sum([1 if n==1 else 0 for n in self.n_comps])
        self.multi_comps = sum([1 if n>1 else 0 for n in self.n_comps])
        print(f"We have {self.single_comps} single comp cutouts and {self.multi_comps} multi")

        # Filter out predicted bboxes that do not cover the focussed pixel
        """
        for name, (x, y), (bboxes, scores) in zip(self.focussed_names, self.focussed_comps, self.pred_bboxes_scores):
            tel = 0
            for bbox, score in zip(bboxes, scores):
                if self.is_within(x*scale_factor,y*scale_factor, bbox[0],bbox[1],bbox[2],bbox[3]):
                    tel+=1

            if tel==0: 
                print(f"{name}, focussed x,y= {x},{y}, all boxes and scores:",list(zip(bboxes,scores)))
                sdfsdf
        """

        pred_central_bboxes_scores = [[(tuple(bbox),score) for bbox, score in zip(bboxes, scores) 
                            if self.is_within(x*scale_factor,y*scale_factor, 
                                bbox[0],bbox[1],bbox[2],bbox[3])] 
                              for (x, y), (bboxes, scores) 
                              in zip(self.focussed_comps, self.pred_bboxes_scores)]
        if debug:
            print("pred_bboxes_scores after filtering out the focussed pixel")
            print(pred_central_bboxes_scores[0])
        # Record for which images we have no central bbox
        self.central_covered = [True if len(bboxes_scores) > 0 else False 
                                      for bboxes_scores in pred_central_bboxes_scores]

        # Take only the highest scoring bbox from this list of bboxes
        
        # Take only the highest scoring bbox from this list of bboxes
        max_score_threshold=0.05
        self.second_best = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[1:4] 
                                      if len(bboxes_scores) > 1 else [[[-1,-1,-1,-1],-99]] 
                                      for bboxes_scores in pred_central_bboxes_scores]
        self.pred_central_bboxes_scores = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[0] 
                                      if len(bboxes_scores) > 0 else [[-1,-1,-1,-1],0] 
                                      for bboxes_scores in pred_central_bboxes_scores]
        #max_scores =[score for bbox,score in self.pred_central_bboxes_scores] 
        assert len(self.unrelated_comps) == len(self.pred_central_bboxes_scores)
        if debug:
            print("pred_bboxes_scores after getting the highest scoring bbox")
            print(self.pred_central_bboxes_scores[0])

        # Check if related source comps fall inside predicted central box
        if self.remove_unresolved:
            self.reinsert_unresolved_for_triplets()

            self.comp_scores = [np.sum([self.is_within(x*scale_factor,y*scale_factor,
                bbox[0],bbox[1],bbox[2],bbox[3]) and not unresolved
                            for unresolved, x,y in list(zip(unresolved_list, xs,ys))])
                            for (xs, ys), unresolved_list, (bbox, score) 
                            in zip(self.related_comps, self.related_unresolved,
                                self.pred_central_bboxes_scores)]

            # Check if unrelated source comps fall inside predicted central box
            self.close_comp_scores = [np.sum([self.is_within(x*scale_factor,y*scale_factor,
                bbox[0],bbox[1],bbox[2],bbox[3]) and not unresolved 
                        for unresolved, x,y in zip(unresolved_list, xs,ys)])
                               for (xs,ys), unresolved_list, (bbox, score) in zip(
                                   self.unrelated_comps, self.unrelated_unresolved,
                                   self.pred_central_bboxes_scores)]
        else:

            self.comp_scores = [np.sum([self.is_within(x*scale_factor,y*scale_factor,
                 bbox[0],bbox[1],bbox[2],bbox[3]) 
                             for x,y in list(zip(comps[0],comps[1]))])
                             for comps, (bbox, score) 
                             in zip(self.related_comps, self.pred_central_bboxes_scores)]

            if debug:
                 print("comp_scores")
                 print(self.comp_scores[0])
                 print('len comp_scores ',len(self.comp_scores))

            # Check if unrelated source comps fall inside predicted central box
            self.close_comp_scores = [np.sum([self.is_within(x*scale_factor,y*scale_factor,
                 bbox[0],bbox[1],bbox[2],bbox[3]) 
                         for x,y in zip(xs,ys)])
                                for (xs,ys), (bbox, score) in zip(self.unrelated_comps,
                                    self.pred_central_bboxes_scores)]

        debug=self.debug
        # 1&2. "Predicted central bbox not existing or misses a number of components" can now be checked
        includes_associated_fail_fraction = self._check_if_pred_central_bbox_misses_comp(debug=debug)

        # 3&4. "Predicted central bbox encompasses too many or too few components" can now be checked
        includes_unassociated_fail_fraction = \
            self._check_if_pred_central_bbox_includes_unassociated_comps(debug=debug)

        print("Plot predictions")
        #if debug or self.save_predictions:
        #    self.plot_predictions(f"all_prediction_debug_images_rgb",cutout_list=list(range(len(self.related_comps))),
        #            debug=False, show_second_best=False, grayscale=False)
        # Save predictions, scores and misboxed categories
        image_source_paths = [p["file_name"] for p in self._predictions[0]]
        source_names = [p.split('/')[-1] for p in image_source_paths]
        image_dest_paths = [os.path.join("all_prediction_debug_images", image_source_path.split('/')[-1])
                for image_source_path in image_source_paths]
        np.savez(os.path.join(self._output_dir,self._dataset_name + "_prediction_statistics.npz"),
                self.focussed_names, image_source_paths, image_dest_paths,
                self.pred_central_bboxes_scores,
                self.second_best, self.misboxed_category)

        return includes_associated_fail_fraction, includes_unassociated_fail_fraction

    def plot_predictions(self, fail_dir_name, imsize=200,cutout_list=None, debug=False,
            lgm_to_kafka=False, show_second_best=False, grayscale=False):
        """Collect ground truth bounding boxes that fail to encapsulate the ground truth pybdsf
        components so that they can be inspected to improve the box-draw-process"""
        #if (self._dataset_name not in [ 'test', 'val','inference']) or cutout_list is None:
        #    return
        from cv2 import imread
        cmap = plt.get_cmap("tab10")

        # Make dir to collect the failed images in
        fail_dir = os.path.join(self._output_dir, self._dataset_name+'_'+fail_dir_name)
        os.makedirs(fail_dir,exist_ok=True)
        # Remove old directory but first check that it contains only pngs
        for f in os.listdir(fail_dir):
            assert f.endswith('.png'), 'Directory should only contain images.'
        for f in os.listdir(fail_dir):
            os.remove(os.path.join(fail_dir,f))

        # Copy debug images to this dir 
        if debug:
            print('misboxed output dir',fail_dir)
        debug=True

        # if code fails here the debug source name or path is probably incorrect
        image_source_paths = [p["file_name"] for p in self._predictions[0]]
        if self.kafka_to_lgm:
            image_source_paths = [p.replace('home/rafael/data','data1') for p in image_source_paths]
            
        source_names = [p.split('/')[-1] for p in image_source_paths]
        image_dest_paths = [os.path.join(fail_dir, image_source_path.split('/')[-1])
                for image_source_path in image_source_paths]
        image_only=False

        if debug:
            try:
                if self.cats == 'dr1':
                    vac_path = '/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.h5'
                    comp_path = '/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.h5'
                    skey = 'Source_Name'
                elif self.cats == 'spring':
                    vac_path = '/data1/mostertrij/data/catalogues/Spring-60-65/sources-v0.5.h5'
                    comp_path = '/data1/mostertrij/data/catalogues/Spring-60-65/components-v0.5.h5'
                    skey = 'Parent_Source'

                vac = pd.read_hdf(vac_path)
                comp_cat = pd.read_hdf(comp_path)

                comp_dict = {s:idx for s, idx in zip(comp_cat.Component_Name.values,comp_cat[skey].values)}
                vac_dict = {s:idx for s, idx in zip(vac.Source_Name.values,vac.index.values)}   
            except:
                pass

        if image_only:

            for src, dest in zip(image_source_paths, image_dest_paths):
                with open(src, 'rb') as fin:
                    with open(dest, 'wb') as fout:
                        copyfileobj(fin, fout, 128*1024)
        else:
            for i in cutout_list:
                if self.remove_unresolved:

                    focus_name, focus_l, rel_l, unrel_l, rel_unresolved, \
                    unrel_unresolved,old_rel_unresolved,  old_unrel_unresolved, (bbox,score), src, dest = self.focussed_names[i], \
                            self.focussed_comps[i], \
                        self.related_comps[i], self.unrelated_comps[i], \
                        self.related_unresolved[i], self.unrelated_unresolved[i], \
                        self.old_related_unresolved[i], self.old_unrelated_unresolved[i], \
                        self.pred_central_bboxes_scores[i], image_source_paths[i], image_dest_paths[i]
                else:
                    focus_name, focus_l, rel_l, unrel_l, (bbox,score), src, dest = self.focussed_names[i], \
                            self.focussed_comps[i], \
                        self.related_comps[i], self.unrelated_comps[i], \
                        self.pred_central_bboxes_scores[i], image_source_paths[i], image_dest_paths[i]

                # Open image 
                #print(src)
                im = imread(src)



                # Plot figure 
                f, ax1 = plt.subplots(1,1, figsize=(8,6))
                # Radio intensity
                if grayscale:
                    ax1.imshow(im[:,:,1], cmap='gray_r')
                    box_col=cmap(0)
                    box_style='dashed'
                    box_width=3
                else:
                    box_width=3
                    box_col='k'
                    box_style='dashed'
                    im[:,:,1] = 255-im[:,:,1]
                    ax1.imshow(im)
                ax1.text(bbox[0],bbox[1]-3,f"{score:.1%}",size=14,color=box_col)
        
                if show_second_best:
                    ax1.text(bbox[0],bbox[1]-3,f"{score:.1%}")
                    for tl, (bbox2, score2) in enumerate(self.second_best[i]):
                        # Second best bounding box
                        color = cmap(tl)
                        if score2 < 0:
                            break
                        if tl==0: ax1.text(bbox2[2]+2,bbox2[1]-3,f"{score2:.1%}", color=color)
                        #if tl==1: ax1.text(bbox2[2]+20,bbox2[3],f"{score2:.1%}", color=color)
                        #if tl==2: ax1.text(bbox2[0],bbox2[3],f"{score2:.1%}", color=color)
                        ax1.plot([bbox2[0],bbox2[2],bbox2[2],bbox2[0],bbox2[0]],
                                np.array([bbox2[1],bbox2[1],bbox2[3],bbox2[3],bbox2[1]]),linestyle='dashed',color=color)
                        break

                    #for tl, (bbox2, score,color) in enumerate(self.second_best[i],cmap[i]):
                    #    # Second best bounding box
                    #    if tl==0: ax1.text(bbox2[2]+20,bbox2[1],f"{score:.1%}", color=color)
                    #    #if tl==1: ax1.text(bbox2[2]+20,bbox2[3],f"{score:.1%}", color=color)
                    #    #if tl==2: ax1.text(bbox2[0],bbox2[3],f"{score:.1%}", color=color)
                    #    ax1.plot([bbox2[0],bbox2[2],bbox2[2],bbox2[0],bbox2[0]],
                    #            np.array([bbox2[1],bbox2[1],bbox2[3],bbox2[3],bbox2[1]]),linestyle='dashed',color=color)

                # Bounding box
                ax1.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],
                        np.array([bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]]),linewidth=box_width, linestyle=box_style,color=box_col)
                #if debug and dest.endswith('ILTJ110530.36+465055.8_radio_DR2_rotated0deg.png'):
                #    print('bbox plotted in debug image:', bbox)
                #    print('predicted bboxes and scores:', self.pred_bboxes_scores[i])

                # Plot optical
                plot_optical=False
                if plot_optical:
                    if 'w1Mag' in cutout.optical_sources.keys():
                        okey = 'w1Mag'
                    else:
                        okey = 'MAG_W1'
                    for w,x,y in zip(cutout.optical_sources[okey], 
                            cutout.optical_sources['x'],cutout.optical_sources['y']):
                        if not np.isnan(w):
                            marker='+'
                            #ax1.text(x,y,f"{w:.2f}",color='y')
                            ax1.plot(x,y,marker=marker,markersize=8,color='y')



                if debug:
                    try:
                        vac_idx = vac_dict[comp_dict[focus_name]]
                        vac_row=vac.iloc[vac_idx]
                        c = SkyCoord(ra=vac_row.RA, dec=vac_row.DEC, unit='deg',frame='icrs')
                        ax1.set_title(c.to_string('hmsdms'), fontsize=16)
                    except:
                        ax1.set_title(focus_name)
                else:
                    ax1.set_title(focus_name)
                # Plot component locations
                if self.inference_only:
                    focus_color = 'lime'
                else:
                    focus_color = 'red'
                ax1.plot(focus_l[0],focus_l[1],marker='s', markersize=9,color=focus_color)
                if self.remove_unresolved:
                    #print("Inside plot_pred, remove unresolved is on")
                    for unresolved,old_unresolved, x,y in zip(rel_unresolved,old_rel_unresolved,rel_l[0],rel_l[1]):
                        if old_unresolved:
                            marker='x'
                        else:
                            marker='.'
                        ax1.plot(x,y,marker=marker,markersize=8,color='r')
                        if unresolved != old_unresolved:
                            ax1.plot(x,y,marker='+',markersize=8,color='r')
                    for unresolved,old_unresolved,x,y in zip(unrel_unresolved,old_unrel_unresolved, unrel_l[0],unrel_l[1]):
                        if old_unresolved:
                            marker='x'
                        else:
                            marker='.'
                        ax1.plot(x,y,marker=marker,markersize=8,color='lime')
                        if unresolved != old_unresolved:
                            ax1.plot(x,y,marker='+',markersize=8,color='lime')
                else:
                    for x,y in zip(rel_l[0],rel_l[1]):
                        ax1.plot(x,y,marker='.',markersize=8,color='r')
                    for x,y in zip(unrel_l[0],unrel_l[1]):
                        ax1.plot(x,y,marker='.',markersize=8,color='lime')


                #if not show_second_best:
                #    ax1.axes.xaxis.set_visible(False)
                #    ax1.axes.yaxis.set_visible(False)    
                d = 40
                ticks = np.arange(0,201,d)
                ax1.set_xticks(ticks)
                ax1.set_yticks(ticks)
                ax1.set_xticklabels([f'{t*1.5/60:.0f}' for t in ticks])
                ax1.set_yticklabels([f'{t*1.5/60:.0f}' for t in ticks])
                ax1.set_xlim(0,200)
                ax1.set_ylim(200,0)
                ax1.set_xlabel('arcmin')
                ax1.set_ylabel('arcmin')
                # Save and close plot
                plt.savefig(dest, bbox_inches='tight')
                plt.close()
                #if grayscale:
                #    print(dest)
                #    sdfdsf

        
    def _check_if_pred_central_bbox_includes_unassociated_comps(self, debug=False):
        """Check whether the predicted central box includes a number of unassocatiated components
            as indicated by the ground truth"""
        # Tally for single comp
        single_comp_fail = [unrelated > 0 for n_comp, unrelated in 
                zip(self.n_comps, self.close_comp_scores) if n_comp == 1]
        single_comp_fail_frac = np.sum(single_comp_fail)/len(single_comp_fail)
            
        # Tally for multi comp
        multi_comp_binary_fail = [unrelated > 0 for n_comp, unrelated in 
                                     zip(self.n_comps, self.close_comp_scores) if n_comp > 1]
        #multi_comp_success = [total for n_comp, total in zip(self.n_comps, self.close_comp_scores) 
        #                            if n_comp > 1]
        multi_comp_binary_fail_frac = np.sum(multi_comp_binary_fail)/len(multi_comp_binary_fail)
        

        # Collect single comp sources that includ unassociated comps
        ran = list(range(len(self.close_comp_scores)))
        fail_indices = [i for i, n_comp, unrelated in 
            zip(ran, self.n_comps, self.close_comp_scores) if n_comp == 1 and unrelated > 0]
        self.misboxed_category[fail_indices] = 2
            
        #if debug or self.save_predictions:
        if 1==2:
            self.plot_predictions("single_overestimation", imsize=200,cutout_list=fail_indices,
                    show_second_best=True, debug=False, lgm_to_kafka=False)

        # Collect single comp sources that fail to include their gt comp
        fail_indices = [i for i, n_comp, unrelated in 
                                 zip(ran, self.n_comps, self.close_comp_scores) 
                                 if n_comp > 1 and unrelated > 0]
        self.misboxed_category[fail_indices] = 4

        #if debug or self.save_predictions:
        if 1==2:
            self.plot_predictions("multi_overestimation",
                    cutout_list=fail_indices, show_second_best=True,  debug=False, lgm_to_kafka=False)
        return single_comp_fail_frac, multi_comp_binary_fail_frac



    def _check_if_pred_central_bbox_misses_comp(self, debug=False):
        """Check whether the predicted central box misses a number of assocatiated components
            as indicated by the ground truth"""

        # Tally for single comp
        single_comp_fail = [not central_covered 
                for n_comp, central_covered in zip(self.n_comps,
            self.central_covered) if n_comp == 1]
        single_comp_fail_frac = np.sum(single_comp_fail)/len(single_comp_fail)
            
        # Tally for multi comp
        multi_comp_binary_fail = [(central_covered and unrelated == 0 and n_comp > (related+1)) or (not central_covered)
                for n_comp, central_covered, related, unrelated in
                zip(self.n_comps, self.central_covered, self.comp_scores, self.close_comp_scores) 
                if n_comp > 1]
        multi_comp_binary_fail_frac = np.sum(multi_comp_binary_fail)/len(multi_comp_binary_fail)
        
        # Collect single comp sources that fail to include their gt comp
        ran = list(range(len(self.n_comps)))
        fail_indices = [i for i, n_comp,central_covered in zip(ran, self.n_comps, self.central_covered) 
                if n_comp == 1 and not central_covered ]

        #if debug or self.save_predictions:
        if 1==2:
            self.plot_predictions("single_missing", cutout_list=fail_indices, show_second_best=True, debug=False, lgm_to_kafka=False)

        self.misboxed_category[fail_indices] = 1
        # Collect single comp sources that fail to include their gt comp
        fail_indices = [i for i, n_comp, central_covered, related, unrelated in
            zip(ran, self.n_comps, self.central_covered, self.comp_scores, 
                self.close_comp_scores) 
            if n_comp > 1 and ((central_covered and unrelated == 0 and n_comp > (related+1)) \
                    or (not central_covered))]
        self.misboxed_category[fail_indices] = 3

        #if debug or self.save_predictions:
        if 1==2:
            self.plot_predictions("multi_underestimation", cutout_list=fail_indices,
                    show_second_best=True, debug=False, lgm_to_kafka=False)

        return single_comp_fail_frac, multi_comp_binary_fail_frac
    


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
