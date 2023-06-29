from typing import Any
import numpy as np

from scipy.stats import wasserstein_distance
from numpy.typing import NDArray
from skimage.segmentation import slic
from typing import Optional


class Slic_Wrapper():
    def __init__(self, 
                 n_segments: int = 10,
                 compactness: float = 10.,
                 sigma: float = 0.):
        self.n_segments= n_segments
        self.compactness = compactness
        self.sigma = sigma

    def __call__(self, image):
        return slic(image, 
                    n_segments=self.n_segments, 
                    compactness=self.compactness, 
                    sigma = self.sigma)
    
    def __repr__(self):
        return f'slic(n_segments={self.n_segments}, compactness={self.compactness}, sigma={self.sigma})'