## Movie review sentiment analysis

Thе scripts in this folder show how to generate a convolutional neural network with PyTorch to predict the sentiment of a movie review.
The data set is the Stanford Sentiment Treebank v2, which contains about 70000 sentences.
For simplicity, we here create a binary classifier: a movie review is either positive or negative.

Based on https://github.com/bentrevett/pytorch-sentiment-analysis
Specifically tutorial 4 -  Convolutional Sentiment Analysis

The notebooks in this folder are organized as follows:

- [prepare_data.ipynb](prepare_data.ipynb) handles the generation of an embedding for all the words present in the training data. This is stored as `word_vectors.txt`.
- [generate_model.ipynb](generate_model.ipynb) defines a neural network and trains it. It depends on the `word_vectors.txt` file.
- [generate_model.py](generate_model.py) is a Python script version of the `generate_model` notebook. It is basically the same, but has the hyperparameters as command-line arguments for easy intergartion with [Weights & Biases](https://wandb.ai).
- [experiments.ipynb](experiments.ipynb) carries out the experiments for Lorentz Workshop.

The `sentiment words distributed.txt` has the words used for the experiments.

In `..model/` we also have:
- The `experiment1.tsv`,`experiment3.tsv`,`experiment3.tsv`, and `experiment4.tsv` files with sentences/adjectives for the explainer/predictor.
- The `wordscore.tsv` has dataset sentiment ratings for the words.
