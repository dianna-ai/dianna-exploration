## Leafsnap model generation

This scripts in this folder show how to generate a convolutional neural network with PyTorch to predict the species from the picture of a leaf in the LeafSnap dataset.

The notebooks in this folder are organized as follows:

- [generate_model.ipynb](generate_model.ipynb) defines a neural network and trains it.
- [generate_model.py](generate_model.py) is a Python script version of the `generate_model` notebook. It is basically the same, but has the hyperparameters as command-line arguments for easy intergartion with [Weights & Biases](https://wandb.ai).