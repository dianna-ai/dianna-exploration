import numpy as np

from typing import Optional, Iterable
from skimage.segmentation import slic
from sklearn.model_selection import ParameterGrid


def create_grid(parameters: object) -> list:
    ''' Convert parameter objects to a grid containing all possible parameter 
        combinations.

        Args:
            parameters: Parameters to use in the grid

        Returns: All possible parameter combinations
    ''' 
    params = parameters.__dict__
    params = {k: params[k] for k in params.keys() 
              if params[k] is not None}
    return list(ParameterGrid(params))


class RISE_parameters(object):
    '''Set up hyperparameters for RISE.
    '''
    def __init__(self, 
                 p_keep: Optional[Iterable] = None, 
                 feature_res: Optional[Iterable] = None, 
                 n_masks: Optional[Iterable] = None,
                 random_state: Optional[Iterable[int]] = None):
        '''
        Args:
            p_keep: probability to keep bit unmasked
            feature_res: size of bitmask
            n_masks: number of masks to use
            random_state: random seed
        '''
        self.p_keep=p_keep
        self.feature_res = feature_res
        self.n_masks = n_masks
        self.random_state = random_state


class LIME_parameters(object):
    '''Set up hyperparamters for LIME. 
    NOTE: LIME segments the image using quickshift which is statically impleneted in 
    their package. We should discuss if we want to make this segmentation modifiable as a 
    hyperparameter by chanigng the LIME implementation and trying out a different segmentation algo.
    '''
    def __init__(self,
                 num_samples: Optional[Iterable] = None,
                 kernel_width: Optional[Iterable] = None,
                 feature_selection: Optional[Iterable] = None,
                 distance_metric: Optional[Iterable] = None,
                 segmentation_fn: Optional[Iterable] = None,
                 model_regressor: Optional[Iterable] = None,
                 random_state: Optional[Iterable] = None):
        '''
        Args:
            num_samples: amount of instances to perturb
            kernel_width: width to use for kernel to compute proximity
            feature_selection: feature selection algorithm to select a priori
            distance_metric: distance metric used to compute proximity
            segmentation_fn: Segmentation algorithm to obtain superpixels
            model_regressor: Surrogate model to use
            random_state: random seed
        '''
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        self.distance_metric = distance_metric 
        self.segmentation_fn = segmentation_fn
        self.model_regressor = model_regressor
        self.random_state = random_state


class SHAP_parameters(object):
    ''' Set up hyperparameters for KernelSHAP.'''
    def __init__(self, 
                 nsamples: Optional[Iterable] = None,
                 background: Optional[Iterable]= None,
                 sigma: Optional[Iterable] = None,
                 l1_reg: Optional[Iterable] = None, 
                 random_state: Optional[Iterable] = None):
        '''
         Args:
            nsamples: amount of combinations to use
            background: background of masked image
            sigma: gaussian kernel width
            l1_reg: L1 regularization factor
            random_state: random seed
        '''
        self.nsamples = nsamples,
        self.background = background
        self.sigma = sigma
        self.l1_reg = l1_reg
        self.random_state = random_state


RISE_config = RISE_parameters(
    p_keep = np.arange(.1, 1, .1),
    feature_res=np.arange(2, 11, 2),
    n_masks=np.arange(1000, 4000, 500)
)


LIME_config = LIME_parameters(
    num_samples=np.arange(1000, 4000, 500),
    kernel_width=np.geomspace(0.01, 3, num=5),
    distance_metric=[None], # will extend later
    segmentation_fn=[slic],
    random_state = [42]
)


SHAP_config = SHAP_parameters(
    nsamples=np.arange(1000, 4000, 500),
    l1_reg=np.geomspace(.001, 1, num=5)
)