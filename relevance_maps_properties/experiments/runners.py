import onnxruntime as ort

from pathlib import Path
from typing import Union, Optional, Callable


class ModelRunner:
    """Runs an onnx model with a set of inputs and outputs."""
    def __init__(self, 
                 filename: Union[str, Path], 
                 preprocess_function: Optional[Callable] = None,
                 device: int = 'cpu'
                ):
        """Generates function to run ONNX model with one set of inputs and outputs.

        Args:
            filename: Path to ONNX model on disk
            preprocess_function: Function to preprocess input data with

        Returns:
            function

        Examples:
            >>> runner = SimpleModelRunner('path_to_model.onnx')
            >>> predictions = runner(input_data)
        """
        self.filename = filename
        self.preprocess_function = preprocess_function
        self.device=device

    def __call__(self, input_data):     
        """Get ONNX predictions."""
        EP_list = self._set_EP()
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(self.filename,  sess_options=sess_options, providers=EP_list)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        if self.preprocess_function is not None:
            input_data = self.preprocess_function(input_data)

        onnx_input = {input_name: input_data}
        pred_onnx = sess.run([output_name], onnx_input)[0]
        return pred_onnx

    def _set_EP(self) -> list:
        ''' Set the inference parameters for onnx

        NOTE: It is vital to use the cudnn_conv_algo_search parameter
            to truly speed up the inference session with the GPU, doing 
            an exhaustive search for convolutions is expensive. 
        '''
        if self.device == 'cpu': # CPU
            return ['CPUExecutionProvider']
        # Importa
        elif self.device == 'gpu': # GPU
            return [("CUDAExecutionProvider", 
                     {"cudnn_conv_algo_search": "DEFAULT"}
                    ),                   
                     "CPUExecutionProvider"
                   ]
        else:
            raise ValueError('Device has to be `cpu` or `gpu`')
