import onnxruntime as ort

from pathlib import Path
from typing import Union, Optional, Callable


class ModelRunner:
    """Runs an onnx model with a set of inputs and outputs."""
    def __init__(self,
                 filename: Union[str, Path],
                 preprocess_function: Optional[Callable] = None,
                 device: int = 0
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
        self.device = device

        EP_list = self._set_EP(device)
        self.sess = ort.InferenceSession(self.filename, providers=EP_list)

    def __call__(self, input_data, device=0):
        """Get ONNX predictions."""

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name

        if self.preprocess_function is not None:
            input_data = self.preprocess_function(input_data)

        onnx_input = {input_name: input_data}
        pred_onnx = self.sess.run([output_name], onnx_input)[0]
        return pred_onnx

    @staticmethod
    def _set_EP(device: int) -> list:
        if device == 0: # CPU
            return ['CPUExecutionProvider']
        elif device == 1: # GPU
            return [("CUDAExecutionProvider",
                     {"cudnn_conv_algo_search": "DEFAULT"}
                    ),
                     "CPUExecutionProvider"
                   ]
        else:
            raise ValueError('Device has to be 0 (CPU) or 1 (GPU).')