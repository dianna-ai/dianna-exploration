import warnings
warnings.filterwarnings('ignore') # disable warnings relateds to versions of tf
import numpy as np
import dianna
import onnx
from onnx_tf.backend import prepare
import matplotlib.pyplot as plt
from pathlib import Path
from dianna import visualization
from keras import utils
import onnx
import onnxruntime
from scipy.special import softmax

def run_model(data, name):
    fname           = str(Path('models', name))
    sess            = onnxruntime.InferenceSession(fname)
    input_name      = sess.get_inputs()[0].name
    output_name     = sess.get_outputs()[0].name

    onnx_input      = {input_name: data}
    pred_onnx       = sess.run([output_name], onnx_input)

    return softmax(pred_onnx[0], axis=1)

DATA_PATH           = Path('data', 'shapes.npz')
MODEL_NAME          = 'geometric_shapes_model.onnx'
MODEL_PATH          = Path('models', MODEL_NAME)
# load dataset
data                = np.load(DATA_PATH)
# load testing data and the related labels
X_test              = data['X_test'].astype(np.float32).reshape([-1, 1, 64, 64])
y_test              = data['y_test']


# Load saved onnx model
onnx_model          = onnx.load(MODEL_PATH)
output_node         = prepare(onnx_model, gen_tensor_dict=True).outputs[0]
pred_onnx           = run_model(X_test, MODEL_NAME)
pred_ids            = pred_onnx.argmax(axis=1)
class_name          = ['circle', 'triangle']

for i_instance in range(len(pred_ids)):
    # select instance for testing
    test_sample     = X_test[i_instance].copy().astype(np.float32)
    # model predictions with added batch axis to test sample
    predictions     = prepare(onnx_model).run(test_sample[None, ...])[f'{output_node}']
    pred_class      = class_name[np.argmax(predictions)]
    print("The predicted class is:", pred_class)
    relevances      = dianna.explain_image(MODEL_PATH, test_sample,
                                                  method="LIME", labels=[pred_ids[i_instance]], nsamples=2000,
                                                  n_masks=1000, feature_res=12, p_keep=0.7,
                                                  axis_labels=('channels','height','width'))

    class_idx       = pred_ids[i_instance]
    fig, ax         = plt.subplots(1,3)
    ax[0].imshow(relevances[0],cmap='jet')
    ax[1].imshow(utils.img_to_array(test_sample[0])/255.,cmap='gray')
    ax[2].imshow(utils.img_to_array(test_sample[0]) / 255., cmap='gray')
    ax[2].imshow(relevances[0], cmap='jet', alpha=0.4)
    plt.title(str(pred_ids[i_instance])+'_'+str(pred_onnx[i_instance,pred_ids[i_instance]]))
    plt.show()
