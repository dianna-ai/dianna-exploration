import dianna

import numpy as np


def get_onnx_names(onnx_model):
    output =[node.name for node in onnx_model.graph.output]
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input, output


def LIME_postprocess(*args, **kwargs):
    results = dianna.explain_image(method='LIME', *args, **kwargs)
    return np.array(results)[0][None, ...]


def SHAP_postprocess(label, *args, **kwargs):
    shapley_values, segments_slic = dianna.explain_image(method='KernelSHAP', *args, **kwargs)
    saliences = list(_fill_segmentation(shapley_values[label][0], segments_slic))
    return np.array(saliences)[np.newaxis, ..., np.newaxis]


def _fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out