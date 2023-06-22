import argparse
import dianna
import quantus
import json
import warnings

import numpy as np

from numpy.typing import NDArray
from onnx import load 
from onnx2keras import onnx_to_keras
from onnx.onnx_ml_pb2 import ModelProto
from pathlib import Path
from tqdm import tqdm
from time import time_ns
from typing import Callable, Union, Optional
from functools import partialmethod

# Local imports
from .hyperparameter_configs import SHAP_config, LIME_config, RISE_config, create_grid
from ..metrics.metrics import Incremental_deletion
from ..metrics import utils

# Silence warnings and tqdm progress bars by default
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
warnings.filterwarnings("ignore")


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
        self.model = dianna.utils.get_function(model, 
                                               preprocess_function=preprocess_function)
        onnx_model = load(model)
        input_names, _ = utils.get_onnx_names(onnx_model)
        self.keras_model = onnx_to_keras(onnx_model, input_names,
                                         name_policy='renumerate', verbose=False)

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

    def init_JSON_format(self, experiment_name: str, 
                         n_images: int, n_configs: int) -> dict:
        ''' Return the hierarchical structure and metadata for the experiments data. 

            Returns the data format that `explain_evaluate_images` expects to dump the 
            results in. 

            Args:
                experiment_name: Name for the experiment
                n_images: Number of images to run the experiment on
                n_configs: Number of hyperparameter configurations
            Returns:
                Base dictionary representing JSON structure as output format.
        '''
        output = {'experiment_name': experiment_name,
                  'images': [
                             {
                              'image_id': 0,
                              'imag_data': [],
                              'configs': [
                                          {
                                           'config_id': 0,
                                           'config': [],
                                           'salient_batch': [],
                                           'incremental_deletion': {},
                                           'avg_sensitivity': 0.,
                                           'max_sensitivity': 0.,
                                           'run_time': 0.,
                                          } for _ in range(n_configs) 
                                         ] 
                             } for _ in range(n_images)
                            ] 
                 }
        return output

    def explain_evaluate_images(self,
                                output_file: Union[str, Path],
                                data: NDArray,
                                method: str,
                                grid: list[dict],
                                batch_size=64,
                                save_between: int = 100,
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
        if output_file.suffix != '.json':
            raise ValueError('`output_file` must end with `.json`.')
        if data.ndim != 4:
            raise ValueError('Dimension of `data` must be 4')

        explainer = self._get_explain_func(method)
        results = self.init_JSON_format(method + 'Experiment', data.shape[0], len(grid))
        run_times = np.empty(self.n_samples)

        salient_batch = np.empty((self.n_samples, *data.shape[1:3]))

        for image_id, image_data in enumerate(tqdm(data, desc='Running Experiments', 
                                                   disable=False, position=0)):
            label = self.model(image_data[np.newaxis, ...],
                        **model_kwargs).argmax()[np.newaxis, ...]
            for config_id, explainer_params in enumerate(tqdm(grid, desc='Trying out configurations',
                                                              disable=False, position=1,
                                                              leave=True)):
                # TODO: Ensure this block happens outside this VERY expensive loop
                explainer_params['labels'] = label
                explainer_params['model_or_function'] = self.model
                explainer_params['input_data'] = image_data

                for i in range(self.n_samples): 
                    start_time = time_ns()
                    salient_batch[i] = explainer(**explainer_params)
                    run_times[i] = time_ns() - start_time

                # Compute metrics
                incr_del = self.incr_del(image_data, 
                                         salient_batch, 
                                         batch_size=batch_size,
                                         **model_kwargs)
                del incr_del['salient_scores']
                del incr_del['random_scores']


                avg_sensitiviy  = self.avg_sensitivity(model=self.keras_model,
                                                       x_batch=image_data[np.newaxis, ...],
                                                       y_batch=label,
                                                       batch_size=batch_size,
                                                       explain_func=explainer,
                                                       explain_func_kwargs=explainer_params)
                max_sensitivity = self.max_sensitivity(model=self.keras_model,
                                                       x_batch=image_data[np.newaxis, ...],
                                                       y_batch=label,
                                                       batch_size=batch_size,
                                                       explain_func=explainer,
                                                       explain_func_kwargs=explainer_params) 
                
                # Save results
                results['images'][image_id]['configs'][config_id]['incremental_deletion'] = incr_del
                results['images'][image_id]['configs'][config_id]['avg_sensitivity'] = avg_sensitiviy
                results['images'][image_id]['configs'][config_id]['max_sensitiviy'] = max_sensitivity
                results['images'][image_id]['configs'][config_id]['run_time'] = np.median(run_times)
                
            # Write imbetween result to file in case of runtime failures
            if image_id % save_between == 0:
                print(f"Backing up at iteration {image_id}")
                with open(output_file, 'w') as fp:
                    json.dump(results, fp)

        # Save final results. 
        with open(output_file, 'w') as fp:
            json.dump(results, fp)

    @staticmethod
    def _get_explain_func(method: str) -> Callable:
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


def pool_handler():
    '''Extend support for distributed computing

       This function should generate several processes such
       that our code can be run in a distributed manner.
    '''
    raise NotImplementedError()


def load_MNIST(data: Union[str, Path]) -> NDArray:
    f_store = np.load(data)
    images = f_store['X_test'].astype(np.float32)
    return images.reshape([-1, 28, 28, 1]) / 255


def main():
    ''' Main function to run the experiments. 

        All experiments are called here, its is configurable through
        command-line arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, default='./')
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=5)

    args = parser.parse_args()
    kwargs = vars(args)
    model = str(Path(kwargs.pop('model')).absolute())
    out = kwargs.pop('out')

    data = load_MNIST(kwargs.pop('data'))
    for method, config in zip(['RISE', 'LIME', 'KernelSHAP'], 
                            [RISE_config, LIME_config, SHAP_config]):
        grid = create_grid(config)
        out = Path(out) / (method + '.json')
        experiments = Experiments(model, **kwargs)
        kwargs = dianna.utils.get_kwargs_applicable_to_function(experiments.explain_evaluate_images, kwargs)
        experiments.explain_evaluate_images(out, data, method, grid, **kwargs)

if __name__ == '__main__':
    main()