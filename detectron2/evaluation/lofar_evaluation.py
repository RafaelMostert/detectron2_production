# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from PIL import Image
from tabulate import tabulate

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

    def __init__(self, dataset_name, output_dir):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        """
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        """

        self._predictions_json = os.path.join(output_dir, "predictions.json")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "file_name":input["file_name"],
                    "focussed_comp":input["focussed_comp"],"related_comp":input["related_comp"],
                    "unrelated_comp":input["unrelated_comp"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)#.numpy()
                prediction["instances"] = instances
            self._predictions.append(prediction)
            """
            #self._inputs.append(input)
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )
            """

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
            self._logger.warning("[LOFAREvaluator] Did not receive valid predictions.")
            return {}

        # Save predicted instances
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)



        includes_associated_fail_fraction, includes_unassociated_fail_fraction, correct_cat = \
            _evaluate_predictions_on_lofar_score(self._dataset_name, self._predictions,
                    self._imsize, self._output_dir, save_appendix=self._dataset_name, scale_factor=self._scale_factor, 
                                        overwrite=self._overwrite, summary_only=True,
                                        comp_cat_path=self._component_cat_path,
                                        fits_dir=self._fits_path, gt_data=self._gt_data,
                                        image_dir=self._image_dir, metadata=self._metadata)

        self._results = OrderedDict()
        self._results["bbox"] = {"assoc_single_fail_fraction": includes_associated_fail_fraction[0],
        "assoc_multi_fail_fraction": includes_associated_fail_fraction[1],
        "unassoc_single_fail_fraction": includes_unassociated_fail_fraction[0],
        "unassoc_multi_fail_fraction": includes_unassociated_fail_fraction[1],
        "correct_catalogue": correct_cat}
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
        """
        #gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        #gt_folder = self._metadata.panoptic_root

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions
            with PathManager.open(self._predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(self._predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)
        """

    def _evaluate_predictions_on_lofar_score(dataset_name, predictions, imsize, output_dir, 
                                        save_appendix='', scale_factor=1, 
                                        overwrite=True, summary_only=False,
                                        comp_cat_path=None, gt_data=None,
                                        fits_dir=None, metadata=None):
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
        predictions = self._predictions
        debug=True
        print("scale_factor", scale_factor)

        ###################### ground truth
        # Get pixel locations of ground truth components
        """
            prediction = {"image_id": input["image_id"], 
            "file_name":input["file_name"], # string of path
                    "focussed_comp":input["focussed_comp"], # single x,y tuple
                    "related_comp":input["related_comp"], # list of xs and list of ys
                    "unrelated_comp":input["unrelated_comp"]} # list of xs and list of ys
        """
        focussed_comps = [p["focussed_comp"] for p in predictions]
        related_comps = [p["related_comp"] for p in predictions]
        unrelated_comps = [p["unrelated_comp"] for p in predictions]
        n_comps = [1+len(c[0]) for c in related_comps]

        gt_locs = (focussed_comps, related_comps, unrelated_comps, n_comps)
        if debug:
            print("ncomps", "locs, centrallocs, closecomplocs")
            print(focussed_comps[0], related_comps[0], n_comps[0])


        ###################### prediction
        # Get bounding boxes per image as numpy arrays
        pred_bboxes_scores = [(image_dict['instances'].get_fields()['pred_boxes'].tensor.numpy(), 
                  image_dict['instances'].get_fields()['scores'].numpy()) 
                 for image_dict in predictions]
        if debug:
            print("pred_bboxes_scores")
            print(pred_bboxes_scores[0])

        # Filter out bounding box per image that covers the focussed pixel
        pred_central_bboxes_scores = [[(tuple(bbox),score) for bbox, score in zip(bboxes, scores) 
                            if is_within(x*scale_factor,imsize-y*scale_factor, 
                                bbox[0],bbox[1],bbox[2],bbox[3])] 
                              for (x, y), (bboxes, scores) in zip(focussed_comps, pred_bboxes_scores)]
        if debug:
            print("pred_bboxes_scores after filtering out the focussed pixel")
            print(pred_central_bboxes_scores[0])
        
        # 1. No predicted box covers the middle pixel
        # can now be checked
        #     fail_fraction_1 = (len(central_bboxes)-len(pred_central_bboxes_scores))/len(central_bboxes)
        #     print(f'{(len(central_bboxes)-len(pred_central_bboxes_scores))} predictions '
        #           f'(or {fail_fraction_1:.1%}) fail to cover the central component of the source.')
        
        # Take only the highest scoring bbox from this list
        pred_central_bboxes_scores = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[0] 
                                      if len(bboxes_scores) > 0 else [[-1,-1,-1,-1],0] 
                                      for bboxes_scores in pred_central_bboxes_scores]

        # Check if other source comps fall inside predicted central box
        #print([loc for loc in locs])
        #print([[x,y for x,y in np.dstack(loc)[0]] for loc in locs])
        # TODO yaxis flip hack
        comp_scores = [np.sum([is_within(x*scale_factor,imsize-y*scale_factor,
            bbox[0],bbox[1],bbox[2],bbox[3]) 
                        for x,y in list(zip(comps[0],comps[1]))])
                                          for comps, (bbox, score) 
                                          in zip(related_comps, pred_central_bboxes_scores)]
        #nana = [(scale_factor*x, scale_factor*y) for x, y in np.dstack(locs[inspect_id])[0]]
        #print("locs scaled", nana)

        # 2. The predicted central box misses a number of components
        # can now be checked
        includes_associated_fail_fraction = _check_if_pred_central_bbox_misses_comp(predictions,
                image_dir, output_dir,
                source_names, n_comps,comp_scores, metadata,gt_data, gt_locs,
                                                    summary_only=summary_only)
        
        # 3. The predicted central box encompasses too many components
        # can now be checked
        if debug:
            print('len comp_scores ',len(comp_scores))
            print('len close comp, pred bbox',len(close_comp_locs),  len(pred_central_bboxes_scores))
        assert len(close_comp_locs) == len(pred_central_bboxes_scores)
        close_comp_scores = [np.sum([is_within(x*scale_factor,imsize-y*scale_factor,
            bbox[0],bbox[1],bbox[2],bbox[3]) 
                    for x,y in zip(xs,ys)])
                            for (xs,ys), (bbox, score) in zip(close_comp_locs, pred_central_bboxes_scores)]
        includes_unassociated_fail_fraction =  _check_if_pred_central_bbox_includes_unassociated_comps(
                predictions, image_dir, output_dir, source_names, n_comps,close_comp_scores, metadata,
                gt_data, gt_locs,
                                                                summary_only=summary_only)
        return includes_associated_fail_fraction, includes_unassociated_fail_fraction


def _print_panoptic_results(pq_res):
    #TODO
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    #TODO
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
