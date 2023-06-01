import os
import dianna 
import dianna 

import numpy as np

from typing import Callable, Optional, Union
from numpy.typing import NDArray
from dianna import utils
from tqdm import tqdm
from sklearn.metrics import auc
from scipy.stats import mode
from copy import copy
from PIL.Image import Image
from torchtext.vocab import Vectors
from matplotlib.figure import Figure

import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)


class Incremental_deletion():
    '''Implemenation for the incremental deletion method. Similar to the evaluation 
    inspired by the RISE paper: https://arxiv.org/pdf/1806.07421.pdf.

    NOTE: 
        - The expected input and ouput of the models varies strongly per implementation. 
          As such providing full portability is tricky, and proceed with caution for the model.
        - Current implementation only handles image data. Extending towards language has proven 
          difficult and likely needs seperate functionality.
    '''
    def __init__(self,
                 model: Union[Callable, str], 
                 step: int, 
                 n_samples: int = 1,
                 preprocess_function: Optional[Callable] = None) -> None:
        ''' 
        Args: 
            model: The model to make predictions and that is to be explained.
            step: The amount of pixels to be deleted per iteration.
            preprocess_function: function to preprocess input for model.
            
        '''
        self.model = utils.get_function(model, preprocess_function=preprocess_function)
        self.n_samples = n_samples
        self.step = step 

    def __call__(self, 
                 input_img: NDArray, 
                 salience_map: NDArray, 
                 batch_size: Optional[int] = None, 
                 n_samples: int = 1,
                 impute_method: Union[NDArray, float, str] = 'channel_mean', 
                 evaluate_random_baseline: bool = True,
                 random_seed: Optional[int] = 0,
                 **model_kwargs) -> dict:
        results = {}
        salient_order = self.get_salient_order(salience_map)
        random_order = self.get_random_order(input_img.shape[:2], random_seed)

        for _ in range(n_samples):
            salient_scores = self.evaluate(input_img, salient_order, batch_size, 
                                           impute_method, **model_kwargs)
            x = np.arange(salient_scores.size) / salient_scores.size
            salient_auc = auc(x, salient_scores)
            if not 'salient_scores' in results: 
                results['salient_scores'] = salient_scores
                results['salient_auc'] = salient_auc
            else: 
                results['salient_scores'] += salient_scores
                results['salient_auc'] += salient_auc
        results['salient_scores'] /= n_samples
        results['salient_auc'] /= n_samples

        if evaluate_random_baseline:
            random_scores = self.evaluate(input_img, random_order, batch_size,
                                            impute_method, **model_kwargs)
            results['random_scores'] = random_scores
            results['random_auc'] = auc(x, random_scores)
        return results
        
    def evaluate(self, 
                 input_img: NDArray, 
                 salient_order: NDArray,
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
        # '''
        if salient_order.shape[0] != np.prod(input_img.shape[:2]) or salient_order.shape[1] != 2:
            salient_order = self.get_salient_order(salient_order)
            raise ValueError(f'Shapes of `salient_order` {salient_order.shape} and \
                              `input_img` {input_img.shape} do not align.')

        impute_value = self._make_impute(input_img, impute_method)
        eval_img = np.copy(input_img) # Perform deletion on a copy

        # Handle models requiring batched input
        if batch_size:
            n_iters = np.prod(input_img.shape[:2]) // (self.step * batch_size)
            scores = np.empty(shape=(n_iters * batch_size + 1))
            pred = self.model(eval_img[None, ...], **model_kwargs)
            gold_lbl = np.argmax(pred)
            scores[0] = pred[:,gold_lbl]
            for k in tqdm(range(n_iters), desc='Evaluating'):
                # Create batch and score model 
                partial_order = salient_order[k * self.step * batch_size: (k + 1) * self.step * batch_size]
                batch = self._create_batch(eval_img, partial_order, impute_value, batch_size)
                score = self.model(batch, **model_kwargs)[:, gold_lbl]
                scores[k * batch_size + 1: (k + 1) * batch_size + 1] = score
        # Handle models without batching
        else: 
            n_iters = np.prod(input_img.shape[:2]) // self.step
            scores = np.empty(shape=(n_iters + 1))
            pred = self.model(eval_img, **model_kwargs)
            gold_lbl = np.argmax(pred)
            scores[0] = np.max(pred, axis=0)[gold_lbl]
            for k in tqdm(range(n_iters), desc='Evaluating'):
                # Delete pixels and score model
                i, j = zip(*salient_order[k * self.step: (k + 1) * self.step]) 
                eval_img[i, j] = impute_value
                score = np.max(self.model(eval_img, **model_kwargs), axis=0)[gold_lbl]
                scores[k + 1] = score
        return scores
        
    def _create_batch(self, eval_img: NDArray, salient_order: NDArray, 
                      impute_value: Union[NDArray, float], batch_size: int) -> NDArray:
        '''Create a batch that can be used by the batched_evaluate method. 
        
        Args: 
            eval_img: Working image to delete pixels on.
            salient_order: Order to delete pixels.
            impute_value: Value to replace to be deleted pixels with.
            batch_size: Size of batch for model inference.
        Raises: 
            AssertionError: If the `salient order` shape does not hold enough values 
                            for the given batch_size 
        Returns:
            Batch that is ready for model inference.
        '''
        assert salient_order.shape[0] == batch_size * self.step

        batch = np.empty(shape=(batch_size, *eval_img.shape), dtype=eval_img.dtype)
        for k in range(batch_size):
            # Delete pixels and add to batch
            i, j = zip(*salient_order[k * self.step: (k + 1) * self.step]) 
            eval_img[i, j] = impute_value
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
            ValueError: If the given input is not an image.
            ValueError: If the given `impute_method` can not be dispatched or its shape 
                        does not constitute that of an image. 
        Returns:
            A value that is used to delete pixels in an image. 
        '''
        if isinstance(impute_method, str): 
            if impute_method == 'channel_mean':
                return np.mean(input_img, axis=(0,1))
            elif impute_method == 'full_mean':
                return float(np.mean(input_img, axis=None))
            elif impute_method == 'channel_median':
                return np.median(input_img, axis=(0,1))
            elif impute_method == 'full_median':
                return float(np.median(input_img, axis=None))
            elif impute_method == 'full_mode':
                return mode(input_img, axis=None)[0]
            else:
                raise ValueError( f"Given method {impute_method} is not supported. Please "
                                    "choose from 'channel_mean', 'full_mean',"
                                    "'channel_median', 'full_median or full_mode.")
        elif isinstance(impute_method, NDArray): 
            if not(impute_method.shape[-1] == 3):
                raise ValueError(f"Shape mismatch for given impute_method value {impute_method}."
                                  "Shape must have dimension (..., 3)")
        return impute_method
    
    @staticmethod
    def get_salient_order(salience_map: NDArray) -> NDArray:
        '''Return the order of relvances in terms of indices of `salience_map`

        Args: 
            salience_map: map of salient scores
        Returns:
            Indices of `salience_map` sorted by their value.
        '''
        return np.stack(np.unravel_index(np.argsort(salience_map, axis=None), 
                                         salience_map.shape), axis=-1)[::-1]
    
    @staticmethod
    def get_random_order(image_shape: tuple, random_seed: Optional[int] = 0) -> NDArray:
        '''Get a random order of coordinates '''
        if isinstance(random_seed, int): 
            np.random.seed(random_seed)

        indices = np.argwhere(np.ones(image_shape)) # Hack to get all cartesian coordinates 
        np.random.shuffle(indices)
        return indices

    @staticmethod
    def visualize(salience_map: NDArray,
                  image_data: NDArray,
                  scores: tuple,
                  labels: tuple,
                  save_to: Optional[str] = None, 
                  show_plot: bool = True,
                  **kwargs) -> Figure:
        '''Visualize the computed scores and its AUC score. 

        Args: 
            scores: The model scores to be visualized
            save_to: path to save the image to
        '''
        fig = dianna.visualization.plot_image(salience_map, image_data, heatmap_cmap='jet', show_plot=False)
        ax1 = fig.axes[0]
        ax2 = fig.add_subplot((1, 0, 1, 1))

        # plot deletion curves
        for score, label in zip(scores, labels): 
            n_steps = score.size
            x = np.arange(n_steps) / n_steps
            # text = 'AUC: {:.3f}'.format(auc(x, scores))
            curve, = ax2.plot(x, score)
            curve.set_label(label)
            ax2.set_xlim(0, 1.)
            ax2.set_ylim(0, 1.05)
            ax2.fill_between(x, 0, score, alpha=.4)
            # ax2.annotate(xy=(.5, .5), va='center', ha='center', text=text, **kwargs)
            ax2.set_title('Model score after removing fraction of pixels', **kwargs)
            ax2.set_xlabel('Fraction of removed pixels', **kwargs)
            ax2.set_ylabel('Model score', **kwargs)
        
        # Force same bbox height and width for axes
        ax1_pos = ax1.get_position()
        ax2_pos = ax2.get_position()
        ax2_pos.y0 = ax1_pos.y0
        ax2_pos.x1 = ax2_pos.x0 + (ax1_pos.x1 - ax1_pos.x0)
        ax2_pos.y1 = ax2_pos.y0 + (ax1_pos.y1 - ax1_pos.y0)
        ax2.set_position(ax2_pos)

        plt.legend()

        # Save or show 
        if show_plot:
            plt.show()
        elif save_to:
            if not save_to.endswith('.png'):
                save_to += '.png'
            plt.save(save_to, dpi=200)

        return fig  


class Single_deletion():
    '''Singly delete elements from a sentence and measure its effect on the cofidence. 
       This method is built for textual data. 
    '''
    def __init__(self, model: Union[Callable, str], 
                 tokenizer: Union[Callable, str], 
                 word_vectors: Union[str, os.PathLike],
                 max_filter_size: Optional[int] = 5, 
                 pad_token: str = '<pad>',
                 unk_token: str = '<unk>') -> None:
        '''
        Args:
            tokenizer: the tokenizer for model inpute.
            word_vectors: path to stored word vectors.
            max_filter_size: the maximum input size for the model
            max_filter_size: the maximum input size for the model
            pad_token: the pad token in vocab
            unk_token: the unk token in vocab
        '''
        self.model = utils.get_function(model, preprocess_function=None)  
        self.tokenizer = utils.get_function(tokenizer, preprocess_function=None)
        self.vocab = Vectors(word_vectors, cache=os.path.dirname(word_vectors))
        self.max_filter_size = max_filter_size
        self.max_filter_size = max_filter_size
        self.pad_token = pad_token
        self.unk_token = unk_token

    def evaluate(self, 
                 salience_map: list[tuple[str, str, float]],
                 input_sentence: str, 
                 impute_value: str = '<unk>',
                 **model_kwargs) -> tuple[NDArray, float]:
        ''' Evaluate a sentence on `model` using the single deletion method.
        Takes `salience_map` and replaces each of the given saliences with
        `impute_value`. Measures the model score on these perturbed sentences 
        and returns the scores for the perturbed sentences and input sentence.

        Args:  
            salience_map: The words, indices and their salient scores.
            input_sentence: Sentence to evaluate.
            impute_value: Value to perturb the sentence with
            model_kwargs: Keywords Arguments for `self.model`
        Returns: 
            Perturbed sentence scores and initial sentence score
        '''
        # Tokenize setence.
        # Tokenize setence.
        tokenized = self._preprocess_sentence(input_sentence)
        eval_sentence = copy(tokenized)
        _, indices, _ = self.sort_salience_map(salience_map)
        _, indices, _ = self.sort_salience_map(salience_map)

        # Get original sentence score.
        init_pred = self.model([eval_sentence], **model_kwargs)
        init_score = init_pred.max()
        init_lbl = init_pred.argmax()
        # Get original sentence score.
        init_pred = self.model([eval_sentence], **model_kwargs)
        init_score = init_pred.max()
        init_lbl = init_pred.argmax()

        impute_value = self.vocab.stoi[impute_value]
        scores = np.empty(len(salience_map))

        for i, token_idx in enumerate(indices):
            # Perturb sentence and score model. 
            # Perturb sentence and score model. 
            tmp = eval_sentence[token_idx]
            eval_sentence[token_idx] = impute_value
            score = self.model([eval_sentence], **model_kwargs).flatten()[init_lbl]
            score = self.model([eval_sentence], **model_kwargs).flatten()[init_lbl]
            eval_sentence[token_idx] = tmp
            scores[i] = score
        return scores, init_score
        return scores, init_score
    
    def _preprocess_sentence(self, input_sentence: str) -> list:
        '''Tokenize and embed sentence. 
            Args: 
                input_sentence: Sentence to be preprocessed. 
            Returns:
                Preprocessed sentence.
        '''
        tokens = self.tokenizer(input_sentence)
        if len(tokens) < self.max_filter_size:
            tokens += [self.pad_token] * (self.max_filter_size - len(tokens))
        if len(tokens) < self.max_filter_size:
            tokens += [self.pad_token] * (self.max_filter_size - len(tokens))
        
        embedded = [self.vocab.stoi[token] if token in self.vocab.stoi 
                    else self.vocab.stoi[self.unk_token] for token in tokens]
        return embedded
    
    def visualize(self, salience_map: tuple[str, str, float], 
                  scores: NDArray,
                  title: Optional[str] = None, 
                  save_to: Optional[str] = None,
                  show_plot: bool = True,
                  **text_kwargs,
                  ) -> None:
        '''Visualize the computed evaluation scores and relevances for the words 
        given in `salience_map`. 

        Args: 
            salience_map: The words, indices and their salient scores.
            scores: Model scores as probabilities. 
            save_to: Path to save visualization to.
            show_plot: Show the plot using visualization backend or not. 
            text_kwargs: Kwargs for the text-based methods of matplotlib.
        '''
        assert len(scores) >= len(salience_map)
        words, indices, relevances = self.sort_salience_map(salience_map)
        words, indices, relevances = self.sort_salience_map(salience_map)

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Word', **text_kwargs)
        ax1.set_ylabel('Relevance', color=color, **text_kwargs)
        ax1.bar(range(len(indices)), relevances, color=color, alpha=.9)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(range(len(indices)), words, rotation=90)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax2.set_ylabel('Probability Drop', color=color ,**text_kwargs)  
        ax2.plot(range(len(indices)), scores, '--o', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

        # force 0 to appear on both axes, comment if don't need
        y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
        y_lims[:, 1] = y_lims[:, 1].clip(0, None)

        # normalize both axes
        y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
        y_lims_normalized = y_lims / y_mags

        # find combined range
        y_new_lims_normalized = np.array([np.min(y_lims_normalized), 
                                          np.max(y_lims_normalized)])

        # denormalize combined range to get new axes
        new_lim1, new_lim2 = y_new_lims_normalized * y_mags
        ax1.set_ylim(new_lim1)
        ax2.set_ylim(new_lim2)

        if title: 
            plt.title(title, **text_kwargs)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        if show_plot:
            plt.show()
        elif save_to:
            if not save_to.endswith('.png'):
                save_to += '.png'
            plt.savefig(save_to, dpi=200)

        return fig

    @staticmethod
    def sort_salience_map(salience_map: list[tuple[str, str, float]], 
                          sort_idx: int = 1) -> tuple:
        ''' Sort a salience map according to the output format of dianna at 
        index `sort_idx`. Expect input format is [(word, word_index, salient_score)]. 
        Additionally, the given salience map is unpacked. 

        Args: 
            salience_map: The words, indices and their salient scores.
            sort_idx: The index of `salience_map` to sort by
        Raises: 
            ValueError: In case sort_idx is out of bounds
        Returns: 
            unpacked and sorted `salience_map`
        '''
        if sort_idx < 0 or not(sort_idx < len(salience_map)):
            raise ValueError(f"sort_idx {sort_idx} out of bounds for given salience map")
        
        ordered = sorted(salience_map, key=lambda x: x[sort_idx])
        words, salient_order, relevance = zip(*ordered)
        return words, salient_order, relevance
    