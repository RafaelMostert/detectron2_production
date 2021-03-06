# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
import pickle
from os import path
import pandas as pd

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def save_obj(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(save_path):
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def inference_on_dataset(model, data_loader, evaluator, overwrite=True, only_zero_rot=True):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()
    
    predictions_save_path = path.join(evaluator._output_dir,
            f'predictions_{evaluator._dataset_name}.pkl')
    if not overwrite and path.exists(predictions_save_path):
        # Load existing predictions if overwrite is false
        print("Loading existing predictions")
        #evaluator._predictions = load_obj(predictions_save_path)
        (evaluator._predictions, evaluator.focussed_comps, evaluator.related_comps, 
            evaluator.unrelated_comps,evaluator.n_comps,evaluator.pred_bboxes_scores,
            evaluator.unrelated_names,evaluator.focussed_names,
            evaluator.related_unresolved, evaluator.unrelated_unresolved, 
            evaluator.wide_focus, evaluator.old_related_unresolved,
            evaluator.old_unrelated_unresolved, evaluator.misboxed_category)  = load_obj(predictions_save_path)
    else:

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                # We only need to evaluate the unrotated images
                #if inputs[0]['file_name'].endswith('ILTJ110530.36+465055.8_radio_DR2_rotated0deg.png'):
                #    print('input filename')
                #    print(inputs[0]['proposals'])

                if only_zero_rot and not inputs[0]['file_name'].endswith('_rotated0deg.png'):
                    continue
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                #missing_box = 'ILTJ123057.73+464446.2_radio_DR2_rotated0deg.png'
                #if inputs[0]['file_name'].endswith(missing_box):
                #    print('output filename',missing_box)
                #    print('inputs:',inputs)
                #    print('outputs:', outputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                # Appends predicted instances to evaluator._predictions 
                evaluator.process(inputs, outputs)

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=10,
                    )
        # Save to pickle
        save_obj([evaluator._predictions,evaluator.focussed_comps,evaluator.related_comps,
            evaluator.unrelated_comps,evaluator.n_comps,evaluator.pred_bboxes_scores,evaluator.unrelated_names,
            evaluator.focussed_names, evaluator.related_unresolved,
            evaluator.unrelated_unresolved, evaluator.wide_focus, evaluator.old_related_unresolved,
            evaluator.old_unrelated_unresolved,evaluator.misboxed_category], 
            predictions_save_path)

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )
    results = evaluator.evaluate()
    if not isinstance(results, pd.DataFrame):
        logger.info(f"LOFAR Evaluation metrics (for all values 0% is best, 100% is worst):")
        logger.info(f"1. Pred. that fail to cover a single comp. source.")
        logger.info(f"{results['bbox']['assoc_single_fail_fraction']:.2%}")
        logger.info(f"2. Pred. that fail to cover all comp. of a " \
                "multi-comp, source.")
        logger.info(f"{results['bbox']['assoc_multi_fail_fraction']:.2%}")
        logger.info(f"3. Pred. that include unassociated comp. for a single comp. source.")
        logger.info(f"{results['bbox']['unassoc_single_fail_fraction']:.2%}")
        logger.info(f"4. Pred. that include unassociated comp. for a " \
                "multi-comp. source.")
        logger.info(f"{results['bbox']['unassoc_multi_fail_fraction']:.2%}")
        logger.info(f"Catalogue is {results['bbox']['correct_catalogue']} correct.")

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
