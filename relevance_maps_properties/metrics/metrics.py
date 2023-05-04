import numpy as np

from typing import Callable, Optional, Union
from nptyping import NDArray
from dianna import utils
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc


import matplotlib.pyplot as plt


class Incremental_deletion():
    '''Implemenation for the incremental deletion method. Similar to the evaluation 
    inspired by the RISE paper: https://arxiv.org/pdf/1806.07421.pdf.

    NOTE: 
        - The expected input and ouput of the models varies strongly per implementation. 
          As such providing full portability is tricky, and proceed with caution for the model.
        - Current implementation only handles image data. Extending towards language has proven 
          difficult and likely needs seperate functionality.
    '''
    def __init__(self, model: Union[Callable, str], step: int, 
                 preprocess_function: Optional[Callable] = None) -> None:
        ''' 
        Args: 
            model: The model to make predictions and that is to be explained.
            step: The amount of pixels to be deleted per iteration.
            preprocess_function: function to preprocess input for model.
            
        '''
        self.model = utils.get_function(model, preprocess_function=preprocess_function)
        self.step = step 
    
    def evaluate(self, input_img: NDArray, salience_map: NDArray,
                 batch_size: Optional[int] = None,
                 impute_method: Union[NDArray, float, str] = 'channel_mean', 
                 **model_kwargs) -> NDArray:
        '''Evaluate a given input image using the incremental deletion. Handles 
        both models that accept only batched input and single instances. 

        Args: 
            input_img: The image to evaluate on. 
            salience_map: The salient scores for input_img given the model.
            batch_size: Batch size to use for model inference. 
            model_kwargs: Keyword arguments specific to the model.
        Returns: 
            The model scores for each iteration of deleted pixels.
        '''
        impute = self._make_impute(input_img, impute_method)
        salient_order = self._get_salient_order(salience_map)
        eval_img = np.copy(input_img) # Perform deletion on a copy

        # Handle models requiring batched input
        if batch_size:
            n_iters = np.prod(eval_img.shape[:2]) // (self.step * batch_size)
            scores = np.empty(shape=(n_iters * batch_size + 1))

            pred = self.model(eval_img[None, ...], **model_kwargs)
            gold_lbl = np.argmax(pred)
            scores[0] = pred[:,gold_lbl]
            for k in tqdm(range(n_iters), desc='Evaluating'):
                # Create batch and score model 
                partial_order = salient_order[k * self.step * batch_size: (k + 1) * self.step * batch_size]
                batch = self._create_batch(eval_img, partial_order, impute, batch_size)
                score = self.model(batch, **model_kwargs)[:, gold_lbl]
                scores[k * batch_size:(k + 1) * batch_size] = score
        # Handle models without batching
        else: 
            n_iters = np.prod(eval_img.shape[:2]) // self.step
            scores = np.empty(shape=(n_iters + 1))
            pred = self.model(eval_img, **model_kwargs)
            gold_lbl = np.argmax(pred)
            scores[0] = np.max(pred, axis=0)[gold_lbl]
            for k in tqdm(range(n_iters), desc='Evaluating'):
                # Delete pixels and score model
                i, j = zip(*salient_order[k * self.step: (k + 1) * self.step]) 
                eval_img[i, j] = impute
                score = np.max(self.model(eval_img, **model_kwargs), axis=0)[gold_lbl]
                scores[k + 1] = score
        return scores
    
    def _get_salient_order(self, salience_map: NDArray) -> NDArray:
        '''Return the order of relvances in terms of indices of `salience_map`

        Args: 
            salience_map: map of salient scores
        Returns:
            Indices of `salience_map` sorted by their value.
        '''
        return np.stack(np.unravel_index(np.argsort(salience_map, axis=None), 
                                         salience_map.shape), axis=-1)[::-1]
    
    def _create_batch(self, eval_img: NDArray, salient_order: NDArray, 
                      impute: Union[NDArray, float], batch_size: int):
        '''Create a batch that can be used by the batched_evaluate method. 
        
        Args: 
            eval_img: Working image to delete pixels on.
            salient_order: Order to delete pixels.
            impute: Value to replace to be deleted pixels with.
            batch_size: Size of batch for model inference.
        Raises: 
            AssertionError: If the `salient order` shape does not hold enough values 
                            for the given batch_size 
        Returns:
            Batch that is ready for model inference.
        '''
        assert salient_order.shape[0] == batch_size * self.step

        batch = np.empty(shape=(batch_size, *eval_img.shape))
        for k in range(batch_size):
            # Delete pixels and add to batch
            i, j = zip(*salient_order[k * self.step: (k + 1) * self.step]) 
            eval_img[i, j] = impute
            batch[k] = eval_img
        return batch

    def _make_impute(self, input_img: NDArray, 
                     impute_method: Union[NDArray, float, str]) -> Union[NDArray, float]:
        '''Dispatcher function to get the value that marks values as deleted. 
        Additionally handles and sanitizes input. 

        Args: 
            input_img: Image to compute imputation values for.
            impute_method: Strategy to choose imputation values for. Can also be 
                           a value proposed by the user. 
        Raises: 
            AssertionError: If the given input is not an image.
            ValueError: If the given `impute_method` can not be dispatched or its shape 
                        does not constitute that of an image. 
        Returns:
            A value that is used to delete pixels in an image. 
        '''
        assert input_img.ndim > 1 and input_img.ndim < 4 # Assert that input is image

        if type(impute_method) is str: 
            if impute_method == 'channel_mean':
                return np.mean(input_img, axis=(0,1))
            elif impute_method == 'full_mean':
                return float(np.mean(input_img, axis=None))
            elif impute_method == 'channel_median':
                return np.median(input_img, axis=(0,1))
            elif impute_method == 'full_median':
                return float(np.median(input_img, axis=None))
            else:
                raise ValueError( f"Given method {impute_method} is not supported. Please "
                                    "choose from 'channel_mean', 'full_mean',"
                                    "'channel_median' or 'full_median.")
        elif type(impute_method) is NDArray: 
            if not(impute_method.shape[-1] == 3):
                raise ValueError(f"Shape mismatch for given impute_method value {impute_method}."
                                  "Shape must have dimension (..., 3)")
        return impute_method

    @staticmethod
    def visualize(scores: NDArray, save_to: Optional[str] = None, **kwargs) -> None:
        '''Visualize the computed scores and its AUC score. 

        Args: 
            scores: The model scores to be visualized
            save_to: path to save the image to
        '''
        n_steps = scores.size
        x = np.arange(n_steps) / n_steps
        text = 'AUC: {:.3f}'.format(auc(x, scores))
        
        plt.plot(x, scores)
        plt.xlim(0, 1.)
        plt.ylim(0, 1.05)
        plt.fill_between(x, 0, scores, alpha=.4)
        plt.annotate(xy=(.5, .5), va='center', ha='center', text=text, **kwargs)
        plt.title('Model score after removing fraction of pixels', **kwargs)
        plt.xlabel('Fraction of removed pixels', **kwargs)
        plt.ylabel('Model score', **kwargs)
        
        # Save or show 
        if save_to:
            plt.savefig(save_to + '.png', dpi=200)
            plt.close()
        else:
            plt.show()


def single_deletion():
    pass


def fidelity_check(model: Callable, salience_map: NDArray, x: NDArray):
    '''
    Use the explanation to mask instance x and feed this masked input 
    to the model. Compares the output of the model given input x and 
    the prediction given x masked with expl.
    '''
    pass


def fidelity(model: Callable, expl: NDArray, X: NDArray, cutoff: float = 0.05) -> float:
    '''
    Use the explanation to mask instance x and feed this masked input 
    to the model using all relevances greater than cutoff.
    Compares the output of the model given input x and the prediction given x masked with salience_map.
    '''
    pass


def stability(model: Callable, x: NDArray, n_samples: int = 10) -> float:
    '''
    Compute the stability of the model for instance x by computing the variance 
    of n_samples queries to the model.
    '''
    pass


def robustness(model: Callable, x: NDArray, n_samples: int = 10) -> float:
    '''
    Generate n_samples perturbations around instance x and compute the variance 
    of these instances. 
    '''
    pass


def target_sensitivity(expl1: NDArray, expl2: NDArray) -> float:
    '''
    Given two explanations compute their similarity score with cosine similarity. 
    Higher similarity implies lower target sensitivity. 
    '''
    pass
