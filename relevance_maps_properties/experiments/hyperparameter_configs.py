from collections.abc import Mapping, Sequence
import numpy as np

from typing import Mapping, Optional, Iterable, Sequence, Union
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge

from .models import Slic_Wrapper


class ParamGrid(ParameterGrid):
    '''Wrapper for ParameterGrid from sklearn.model_selection'''
    def __init__(self, param_grid: Union[Sequence, Mapping]) -> None:
        cleaned_grid = {}
        for key in param_grid:
            if param_grid[key] is None:
                continue
            elif isinstance(param_grid[key], (np.ndarray, np.generic)):
                cleaned_grid[key] = param_grid[key].tolist()
            else:
                cleaned_grid[key] = param_grid[key]
        super().__init__(cleaned_grid)
    
    def __getitem__(self, ind: int) -> dict[str, list[str, int, float]]:
        '''Slight modifitcation of the sklearn.model_selection.ParameterGrid implementation
           
           Tries to get the representation of non strings, floats and ints in order 
           to make this data serializable.'''
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    val = v_list[offset]
                    if not isinstance(val, (str, float, int)):
                        val = str(val)
                    out[key] = val
                return out

        raise IndexError("ParameterGrid index out of range")

        
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
                 num_features: Optional[Iterable] = None,
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
        self.num_features = num_features
        self.segmentation_fn = segmentation_fn
        self.model_regressor = model_regressor
        self.random_state = random_state


class SHAP_parameters(object):
    ''' Set up hyperparameters for KernelSHAP.'''
    def __init__(self, 
                 nsamples: Optional[Iterable] = None,
                 n_segments: Optional[Iterable] = None,
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
        self.nsamples = nsamples
        self.n_segments = n_segments
        self.background = background
        self.sigma = sigma
        self.l1_reg = l1_reg
        self.random_state = random_state


RISE_config = RISE_parameters(
    n_masks=np.arange(400, 2000, 200),
    p_keep = np.arange(.05, 1, .05),
    # feature_res=np.arange(3, 16, 3),
    random_state=[42]
)


LIME_config = LIME_parameters(
    num_samples=np.arange(400, 2000, 400),
    kernel_width=np.geomspace(10, 100, num=5),
    # distance_metric=None, # will extend later
    # segmentation_fn=[Slic_Wrapper(n_segments=n) for n in range(10, 60,10)],
    num_features = [None],
    feature_selection=['none'],
    model_regressor=[Ridge(alpha=a) for a in [0, *np.geomspace(0.05, 3, num=4)]]
    # random_state = [42]
)


SHAP_config = SHAP_parameters(
    nsamples=np.arange(60, 270, 40),
    n_segments = np.arange(10, 60, 10),
    l1_reg=[0.1, *np.geomspace(0.5, 3, num=4)],
    random_state=[42]
)
