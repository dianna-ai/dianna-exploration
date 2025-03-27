import argparse
import json
import warnings
from copy import copy
from functools import partialmethod
from pathlib import Path
from time import time_ns
from typing import Callable, Optional, Union

from sklearn.linear_model import LinearRegression, Ridge

import dianna
import numpy as np
import pandas as pd
import quantus
import torch
from dianna.utils.tokenizers import SpacyTokenizer
from numpy.typing import NDArray
from onnx import load
from onnx.onnx_ml_pb2 import ModelProto
from onnx2keras import onnx_to_keras
from quantus import normalise_func
from tqdm import tqdm
from torchtext.vocab import Vectors

# Local imports
from ..metrics import utils
from ..metrics.metrics import Incremental_deletion, Single_deletion
from ..metrics.sensitivity import Sensitivity
from .hyperparameter_configs import LIME_config, RISE_config, SHAP_config, ParamGrid  # noqa: F401
from .runners import ModelRunner
from .models import Model, Predictor  # noqa: F401

# Silence imported progress bars and warnings
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
warnings.simplefilter("ignore")


class Experiments(object):
    '''Class for the hyperparamter experiments. 

       All the necessary functionality with regards to the experiments is implemented
       here. 
       
       NOTE: This method utilizes JSON as a means to store data, however, with the 
       data possibily scaling up to large size, we should look into mongoDB backend 
       or HD5 file storage.
    '''
    def __init__(self, 
                 model: Union[ModelProto, str, Path],
                 n_samples: int = 5,
                 preprocess_function: Optional[Callable] = None,
                 **kwargs):
        '''
        Args:
            model: the black-box model
            n_samples: Number of samples to use for evaluation 
            preprocess_function: Preprocess function for the model
            evaluator_kwargs: Kwargs for evaluation methods
            model_kwargs, kwargs for the black-box model
        Raises:
            TypeError: In case model type mismatched with expected tpyes
        '''
        # Model preprocessing for cross-framework evaluation
        self.str_model = model
        self.model = ModelRunner(model, preprocess_function=preprocess_function)
        onnx_model = load(model)
        input_names, _ = utils.get_onnx_names(onnx_model)
        try:
            self.keras_model = onnx_to_keras(onnx_model, input_names,
                                         name_policy='renumerate', verbose=False)
        except Exception as e:
            raise Exception(e)

        self.n_samples = n_samples
        id_kwargs = dianna.utils.get_kwargs_applicable_to_function(
                    Incremental_deletion.__init__, kwargs)
        quantus_kwargs = dianna.utils.get_kwargs_applicable_to_function(
                         quantus.AvgSensitivity.__init__,kwargs)

        self.incr_del = Incremental_deletion(self.model, **id_kwargs)
        self.avg_sensitivity = quantus.AvgSensitivity(nr_samples=self.n_samples, 
                                                      **quantus_kwargs)
        self.max_sensitivity = quantus.MaxSensitivity(nr_samples=self.n_samples, 
                                                      **quantus_kwargs)

    @staticmethod
    def init_JSON_format(experiment_name: str, 
                         n_configs: int) -> dict:
        ''' Return the hierarchical structure and metadata for the experiments data. 

            Returns the data format that `explain_evaluate_images` expects to dump the 
            results in. 

            Args:
                experiment_name: Name for the experiment
                n_configs: Number of hyperparameter configurations
            Returns:
                Base dictionary representing JSON structure as output format.
        '''
        output = {'experiment_name': experiment_name,
                  'image_id': 0,
                  'image_data': [],
                  'model_scores': [],
                  'configs': [
                                {
                                'config_id': 0,
                                'config': [],
                                'sensitivity': [],
                                'run_time': 0.,
                                'incremental_deletion': {},
                                'salient_batch': [],
                                } for _ in range(n_configs) 
                            ] 
                 } 
        return output

    def explain_evaluate_images(self,
                                output_folder: Union[str, Path],
                                data: NDArray,
                                method: str,
                                grid: list[dict],
                                device: str = 'cpu',
                                batch_size=64,
                                model_kwargs: dict = {}
                                ) -> None:
        ''' This function will run our explainers and evaluators. 

            Args:
                output_file: File to write the results to.
                data: The image data to experiment on
                method: The explainer method to use
                grid: The grid of possible hyperparameter configurations
                save_between: Save results for every save_between images
                model_kwargs: Kwargs to use for the model
        '''
        if data.ndim != 4:
            raise ValueError('Dimension of `data` must be 4')

        explainer = self.get_explain_func(method)
        results = self.init_JSON_format(method + '_Experiment', len(grid))
        run_times = np.empty(self.n_samples)
        salient_batch = np.empty((self.n_samples, *data.shape[1:3]))

        for image_id, image_data in enumerate(tqdm(data, desc='Running Experiments', 
                                                   disable=False, position=0)):
            label = self.model(image_data[np.newaxis, ...],
                        **model_kwargs).argmax()[np.newaxis, ...]
            for config_id, config in enumerate(tqdm(grid, desc='Trying out configurations',
                                                              disable=False, position=1,
                                                              leave=True)):
                
                explainer_params = copy(config) # Prevent in-place modification of grid
                explainer_params['labels'] = label
                explainer_params['model_or_function'] = self.str_model
                explainer_params['input_data'] = image_data
                explainer_params['batch_size'] = batch_size

                for i in range(self.n_samples): 
                    start_time = time_ns()
                    salient_batch[i] = explainer(**explainer_params)
                    run_times[i] = time_ns() - start_time

                # Compute metrics
                incr_del = self.incr_del(image_data, 
                                         salient_batch, 
                                         impute_method='full_mode',
                                         batch_size=batch_size,
                                         **model_kwargs)
                sensitivity = self.avg_sensitivity(model=self.keras_model,
                                                    x_batch=image_data[np.newaxis, ...],
                                                    y_batch=label,
                                                    batch_size=batch_size,
                                                    explain_func=explainer,
                                                    explain_func_kwargs=explainer_params)
                                                    
                # Save results
                results['configs'][config_id]['config'] = grid[config_id]
                results['configs'][config_id]['salient_batch'] = salient_batch.tolist()
                results['configs'][config_id]['incremental_deletion'] = incr_del
                results['configs'][config_id]['sensitivity'] = sensitivity
                results['configs'][config_id]['run_time'] = np.median(run_times)

            # Savel results
            output_file = Path(output_folder) / ('image_' + str(image_id + 87) + '.json')
            with open(output_file, 'w') as fp:
                json.dump(results, fp, indent=4)

    def get_explain_func(self, method: str) -> Callable:
        '''Helper func to return appropriate explain function for method.
        
           Args:
               method: Name of explanation method
           Returns:
               A function that contians the explanation method with post-processing
        '''
        if not isinstance(method, str):
            raise TypeError('Please provide `method` as type str')
        
        if method.upper() == 'KERNELSHAP':
            return utils.SHAP_postprocess
        elif method.upper() == 'LIME':
            return utils.LIME_postprocess
        elif method.upper() == 'RISE':
            return utils.RISE_postprocess
        else: 
            raise ValueError('''Given method is not supported, please choose between
                                KernelShap, RISE and LIME.''')


class TextExperiments(Experiments):
    def __init__(self, 
                 model: Predictor,
                 torch_model: torch.nn.Module,
                 tokenizer: Union[Callable, str, None],
                 word_vectors: Union[str, Path, None],
                 n_samples: int = 25,
                 preprocess_function: Optional[Callable] = None,
                 **kwargs):
        self.model = model
        self.torch_model = torch_model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.vocab = Vectors(word_vectors)
        self.single_del = Single_deletion(model, tokenizer.tokenize, word_vectors)
        self.sensitivity = Sensitivity(utils.Embedder(self.vocab), 
                                       nr_samples=self.n_samples,
                                       perturb_func=utils.Synonym_replacer(),
                                       normalise=True,
                                       normalise_func=normalise_func.normalise_by_average_second_moment_estimate)

    def explain_evaluate_text(self,
                              output_folder: Union[str, Path],
                              data: NDArray,
                              method: str,
                              grid: list[dict],
                              batch_size=64,
                              model_kwargs: dict = {}):
        explainer = self.get_explain_func(method)
        results = self.init_JSON_format(method + '_Experiment', len(grid))

        for text_id, text_data in enumerate(tqdm(data, desc='Running Experiments', 
                                                   disable=False, position=0)):
            label = self.model([text_data], **model_kwargs).argmax()[np.newaxis, ...]
            tokenized = np.array(self.tokenizer.tokenize(text_data))
            for config_id, config in enumerate(tqdm(grid, desc='Trying out configurations',
                                                              disable=False, position=1,
                                                              leave=True)):
                salient_batch = []
                explainer_params = copy(config) # Prevent in-place modification of grid
                explainer_params['labels'] = label
                explainer_params['model_or_function'] = self.model
                explainer_params['input_text'] = text_data
                explainer_params['tokenizer'] = self.tokenizer
                explainer_params['batch_size'] = batch_size
                explainer_params['modality'] = 'text' 

                start_time = time_ns()
                for _ in range(self.n_samples):
                    salience_map = dianna.explain_text(method=method, **explainer_params)[0]
                    for i, saliency in enumerate(salience_map):
                        salience_map[i] = (*saliency[:-1], float(saliency[-1]))
                    salient_batch.append(salience_map)
                run_time = time_ns() - start_time

                # Compute metrics
                single_del = self.single_del(text_data,
                                             salient_batch, 
                                             normalise=True, 
                                             normalise_fn=normalise_func.normalise_by_average_second_moment_estimate,
                                             **model_kwargs)
                sensitivity = self.sensitivity(model=self.torch_model,
                                               x_batch=tokenized[np.newaxis, ...],
                                               y_batch=label,
                                               batch_size=batch_size,
                                               explain_func=explainer,
                                               explain_func_kwargs=explainer_params)
                                                
                # Save results
                results['configs'][config_id]['config'] = grid[config_id]
                results['configs'][config_id]['salient_batch'] = salient_batch
                results['configs'][config_id]['incremental_deletion'] = single_del
                results['configs'][config_id]['sensitivity'] = sensitivity
                results['configs'][config_id]['run_time'] = run_time

            # Savel results
            output_file = Path(output_folder) / ('review_' + str(text_id) + '.json')
            with open(output_file, 'w') as fp:
                json.dump(results, fp, indent=4)


def load_MNIST(data: Union[str, Path]) -> NDArray:
    images = np.load(data)
    images = images.reshape([-1, 28, 28, 1]) / 255
    return images.astype(np.float32)[87:]


def load_movie_review(data: Union[str, Path], tokenizer: Callable) -> NDArray:
    return np.array(pd.read_csv(data, delimiter='\t')['sentence'])[95:]


def main():
    ''' Main function to run the experiments. 

        All experiments are called here, its is configurable through
        command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--word_vectors', type=str)
    parser.add_argument('--out', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    kwargs = vars(args)
    
    modality = kwargs.pop('modality')
    out = Path(kwargs.pop('out'))
    if not out.exists():
        raise ValueError('Please specify an existing path on the --out parameter')
    model = str(Path(kwargs.pop('model')).absolute())
    method = kwargs.pop('method') 
    grid = ParamGrid(globals()[method[-4:] + '_config'].__dict__)

    if modality == 'text':
        if not kwargs['word_vectors']:
            raise ValueError('Please provide `--word_vectors` command-line arg for `--modality text.`')
        kwargs['tokenizer'] = SpacyTokenizer()

    elif modality =='image':
        data = load_MNIST(kwargs.pop('data'))
        experiments = Experiments(model, **kwargs)
        kwargs = dianna.utils.get_kwargs_applicable_to_function(experiments.explain_evaluate_images,  kwargs)
        experiments.explain_evaluate_images(out, data, method, grid, **kwargs)
    
    else:
        raise ValueError(f'Modality {modality} is not implemented, \
                                    please choose between `text` or `image`.')

if __name__ == '__main__':
    main()