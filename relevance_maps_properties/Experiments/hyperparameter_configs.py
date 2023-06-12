from typing import Optional, Callable
import numpy as np

from sklearn.model_selection import ParameterGrid
from numpy.typing import NDArray
from dataclasses import dataclass


def create_grid(parameters: object) -> list:
    ''' Convert parameter objects to a grid containing all possible parameter 
        combincations.

        Args:
            parameters: Parameters to use in the grid

        Returns: All possible parameter combinations
    '''
    return list(ParameterGrid(parameters.__dict__))


class RISE_parameters(object):
    '''Set up hyperparameters for RISE.'''
    def __init__(self, 
                 p_keep: Optional[NDArray] = None, 
                 feature_res: Optional[NDArray] = None, 
                 n_masks: Optional[NDArray] = None):
        '''
        
        '''
        self.p_keep=p_keep
        self.feature_res = feature_res
        self.n_masks = n_masks


class LIME_parameters(object):
    '''Set up hyperparamters for LIME. 
    NOTE: LIME segments the image using quickshift which is statically impleneted in 
    their package. We should discuss if we want to make this segmentation modifiable as a 
    hyperparameter by chanigng the LIME implementation and trying out a different segmentation algo.
    '''
    def __init__(self,
                 num_samples: Optional[NDArray] = None,
                 kernel_width: Optional[NDArray] = None,
                 feature_selection: Optional[NDArray] = None,
                 distance_metric: Optional[list[str]] = None,
                 model_regressor: Optional[list[object]] = None,
                 random_state: Optional[list[int]] = None):
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        self.distance_metric = distance_metric # I'd like to use the Wasserstein distance for image data.
        self.model_regressor = model_regressor
        self.random_state = random_state


class SHAP_parameters(object):
    ''' Set up hyperparameters for KernelSHAP.'''
    def __init__(self, 
                 nsamples: Optional[NDArray] = None,
                 background: Optional[NDArray]= None,
                 sigma: Optional[NDArray] = None):
        self.nsamples = nsamples,
        self.background = background
        self.sigma = sigma

