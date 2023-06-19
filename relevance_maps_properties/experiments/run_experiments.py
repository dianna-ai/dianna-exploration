import argparse
import dianna
import quantus
import json

import numpy as np

from dianna.utils.onnx_runner import SimpleModelRunner
from multiprocessing import Process
from numpy.typing import NDArray
from onnx import load 
from onnx2keras import onnx_to_keras
from onnx.onnx_ml_pb2 import ModelProto
from pathlib import Path
from tqdm import tqdm
from time import time_ns
from typing import Callable, Union, Optional

# Local imports
from .hyperparameter_configs import SHAP_config, LIME_config, RISE_config, create_grid
from ..metrics.metrics import Incremental_deletion
from ..metrics import utils


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
                 evaluator_kwargs: Optional[dict] = None,
                 model_kwargs: Optional[dict] = None,
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
        if isinstance(model, (str, Path)):
            model = load(model)
        if isinstance(model, ModelProto):
            self.model = dianna.utils.get_function(model, preprocess_function=preprocess_function)
            input_names, _ = utils.get_onnx_names(self.model)
            self.keras_model = onnx_to_keras(self.model, input_names)
        else:
            raise TypeError('`model_or_function` failed to convert to Keras.')

        self.n_samples = n_samples
        id_kwargs = dianna.utils.get_kwargs_applicable_to_function(Incremental_deletion.__init__, evaluator_kwargs)
        quantus_kwargs = dianna.utils.get_kwargs_applicable_to_function(quantus.AvgSensitivity.__init__, evaluator_kwargs)

        self.incr_del = Incremental_deletion(self.model, **id_kwargs, **model_kwargs)
        self.avg_sensitivity = quantus.AvgSensitivity(nr_samples=self.n_samples, 
                                                      **quantus_kwargs)
        self.max_sensitivity = quantus.MaxSensitivity(nr_samples=self.n_samples, 
                                                      **quantus_kwargs)

    def init_JSON_format(experiment_name: str, n_images: int, n_configs: int) -> dict:
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
                  'image': [
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
                                          } 
                                         ] * n_configs
                             }
                            ] * n_images
                 }
        return output

    def explain_evaluate_images(self,
                                output_file: Union[str, Path],
                                data: NDArray,
                                method: str,
                                grid: list[dict],
                                save_between: int = 100,
                                model_kwargs: Optional[dict] = None
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

        explainer = self._get_explain_func(method)
        results = self.init_JSON_format(data.shape[0], len(grid))
        
        for image_id, image_data in enumerate(tqdm(data, desc='Running Experiments')):
            results['images'][image_id]
            for config_id, explainer_params in enumerate(grid):
                results['runs']['image_id'][image_id]['params_id'] = {}
                salient_batch = np.empty((self.n_samples, *image_data.shape[:2]))

                start_time = time_ns()
                for i in range(self.n_samples): 
                    salient_batch[i] = explainer(image_data, **explainer_params)
                end_time = (time_ns() - start_time) / self.n_samples

                # Compute metrics
                y_batch = self.model(image_data, **model_kwargs).argmax()[np.newaxis, ...]
                incr_del = self.incr_del(image_data, 
                                         salient_batch, 
                                         batch_size=self.batch_size,
                                         **model_kwargs).pop('salient_batch')
                avg_sensitiviy  = self.avg_sensitivity(model=self.keras_model,
                                                       x_batch=salient_batch,
                                                       y_batch=y_batch,
                                                       batch_size=self.batch_size)
                max_sensitivity = self.max_sensitivity(model=self.keras_model,
                                                       x_batch=image_data,
                                                       y_batch=y_batch,
                                                       batch_size=self.batch_size) 
                
                # Save results
                results['images'][image_id]['configs'][config_id]['incremental_deletion'] = incr_del
                results['images'][image_id]['configs'][config_id]['avg_sensitivity'] = avg_sensitiviy
                results['images'][image_id]['configs'][config_id]['max_sensitiviy'] = max_sensitivity
                results['run_time'] = end_time - start_time
                
                # Write imbetween result to file in case of runtime failures
                if image_id % save_between == 0:
                    print(f"Backing up at iteration {image_id}")
                    with open(output_file, 'w') as f_out:
                        json.dumps(results, f_out)

        # Save final results. 
        with open(output_file, 'w') as f_out:
            json.dumps(results, f_out)

    def _get_explain_func(method: str) -> Callable:
        '''Helper func to return appropriate explain function for method.
        
           Args:
               method: Name of explanation method
           Returns:
               A function that contians the explanation method with post-processing
        '''
        if not isinstance(method, str):
            raise TypeError('Please provide `method` as type str')
        
        if method.to_upper() == 'KERNELSHAP':
            return utils.SHAP_postprocess
        elif method.to_upper() == 'LIME':
            return utils.LIME_postprocess
        elif method.to_upper() == 'RISE':
            return dianna.explain_image
        else: 
            raise ValueError('''Given method is not supported, please choose between
                                KernelShap, RISE and LIME.''')


def pool_handler():
    '''Extend support for distributed computing

       This function should generate several processes such
       that our code can be run in a distributed manner.
    '''
    raise NotImplementedError()


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

    data = np.load(kwargs.pop('data'))
    for method, grid in zip(['RISE', 'LIME', 'KernelSHAP'], 
                            [RISE_config, LIME_config, SHAP_config]):
        out = Path(kwargs.pop('out') / method / '.json')
        experiments = Experiments(kwargs.pop('model') **kwargs)
        proc = Process(target=experiments.explain_evaluate_images, 
                       args=[out, data, method], kwargs=kwargs)
        proc.start

if __name__ == '__main__':
    main()