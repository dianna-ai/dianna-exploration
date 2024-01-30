import json
import pickle

import numpy as np
import distance_explainer.distance
from tensorflow.keras.applications.vgg16 import VGG16
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from notebooks_for_distance_explainer.nice_picture_mprt import NicePicturesMPRT

warnings.filterwarnings('ignore')  # disable warnings related to versions of tf
import numpy as np


def main(n_masks=1000):
    def load_img(path, target_size):
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    model = VGG16()

    input_title = 'bee'
    bee_img, bee_arr = load_img('data/' + input_title + '.jpg', (224, 224, 3))
    reference_title = 'bee2'
    reference_img, reference_arr = load_img(Path('data/' + reference_title + '.jpg'), (224, 224, 3))
    embedded_reference = model(reference_arr)

    masks = distance_explainer.distance.generate_masks_for_images(bee_arr.shape[1:3], n_masks, 0.5, 8)

    channel_first = True  # transpose to always have channels first (pytorch style)

    def explain(model, inputs, targets, **kwargs) -> np.ndarray:
        """

        :param model:
        :param inputs:
        :param targets: Ignored, added because of required signature (source: https://captum.ai/api/gradient_shap.html#captum.attr.GradientShap.attribute)
        :param kwargs:
        :return:
        """

        if channel_first:
            inputst = inputs.transpose([0, 2, 3, 1])
        else:
            inputst = inputs

        # see C:\Users\ChristiaanMeijer\anaconda3\envs\distance_explainer311\Lib\site-packages\quantus\functions\explanation_func.py
        batch_size = inputst.shape[0]
        saliencies = np.empty(inputst.shape)
        for i in range(batch_size):
            saliencies[i], _ = distance_explainer.distance.DistanceExplainer(
                axis_labels=['x', 'y', 'channels'],
                n_masks=n_masks).explain_image_distance(model, inputst[i], embedded_reference, masks=masks)

        # See quantus\metrics\base.py:l425 Channels first!
        if channel_first:
            return saliencies.transpose([0, 3, 1, 2])
        else:
            return saliencies

    attribution_maps_path = Path('_'.join(['attribution_maps', str(n_masks), input_title, reference_title]) + '.pk')
    if not attribution_maps_path.exists():
        onze_MPRT = NicePicturesMPRT()
        for i in range(1):
            onze_MPRT(
                model=model,
                x_batch=(bee_arr.transpose([0, 3, 1, 2])),
                y_batch=[0],
                a_batch=None,
                channel_first=channel_first,
                explain_func=explain,
            )
        axis_0 = 0
        n_layers = range(len(onze_MPRT.a_instance_perturbed_output))
        attribution_maps = {layer: onze_MPRT.a_instance_perturbed_output[layer][axis_0] for layer in n_layers}
        with open(attribution_maps_path, 'wb') as f:
            pickle.dump(attribution_maps, f)

    with open(attribution_maps_path, 'rb') as f:
        attribution_maps = pickle.load(f)

    plot_N_layers = 16
    n_columns = 6
    n_rows = int(np.ceil((plot_N_layers + 1) / n_columns))
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(16, 10))
    plot_step_layers = 1

    def flatten(xss):
        return [x for xs in xss for x in xs]

    layer_set = range(0, plot_step_layers * (plot_N_layers + 1), plot_step_layers)
    for ax_ix, layer in enumerate(layer_set):
        flatten(axs)[ax_ix].imshow(attribution_maps[layer], cmap='Greys')


if __name__ == '__main__':
    main()
