import numpy as np

from scipy.stats import wasserstein_distance
from numpy.typing import NDArray


def wasserstein_RGB(image: NDArray) -> float:
    if not (image.ndim == 3 and image.shape[-1] == 3):
        raise ValueError(f'image must be of shape (Any, Any, 3), you provided: {image.shape}')
    
    distances = np.apply_along_axis(wasserstein_distance, 2, image)
    return distances.mean(axis=None)

