import dianna

import numpy as np

from dianna.methods.lime import LIMEImage
from onnx.onnx_ml_pb2 import ModelProto
from numpy.typing import NDArray
from typing import Union


def get_onnx_names(onnx_model: ModelProto) -> tuple:
    '''
    Gets the names of the input and output layers used to save an onnx model.

    Args:
        onnx_model: The model to extract the names out of.
    Returns:
        net_feed_input, output: names used for the input and output layers.
    '''
    output =[node.name for node in onnx_model.graph.output]
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input, output


def LIME_postprocess(*args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA LIME in according to what Quantus expects. 

    DIANNA yields: list[NDArray[(Any, Any), Any]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    init_kwargs = dianna.utils.get_kwargs_applicable_to_function(LIMEImage.__init__, 
                                                                 kwargs)
    explainer = LIMEImage(**init_kwargs)
    results = explainer.explain(*args, method='LIME', return_masks=False, **kwargs)
    return np.array(results)


def SHAP_postprocess(*args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA KernelSHAP in according to what Quantus expects. 

    DIANNA yields: tuple[NDArray[(Any, Any), Any], NDArray[(Any, Any), Any]]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    shapley_values, segments_slic = dianna.explain_image(method='KernelSHAP', *args, **kwargs)
    saliences = list(_fill_segmentation(shapley_values[0], segments_slic))
    return np.array(saliences)[np.newaxis, ..., np.newaxis]

def RISE_postprocess(*args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA LIME in according to what Quantus expects. 

    DIANNA yields: list[NDArray[(Any, Any), Any]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    results = dianna.explain_image(*args, method='RISE', **kwargs)
    return np.array(results)


def _fill_segmentation(values: NDArray, segmentation: NDArray) -> NDArray:
    '''
    Helper function to mask a segmentation with Shapeley Values

    Args:
        values: Shapeley values
        segmentation: the indices where the shapeley values reside
    Returns:
        The segmented Shapeley values
    '''
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out