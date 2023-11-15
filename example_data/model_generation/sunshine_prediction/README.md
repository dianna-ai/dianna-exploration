## Sunshine hours prediction regression model generation

Thе scripts in this folder show how to generate a 3-layer convolutional neural network with Keras to predict tomorrow's sunshine hours based on meteorological information from today. The data is from the light version of the weather prediction dataset which can be [downloaded from Zenodo](https://doi.org/10.5281/zenodo.5071376). The data contains daily weather observations for 11 different European cities including variables such as 'mean temperature', 'max temperature', etc for the years from 2000 to 2010. For this regression model, the data from Basel will be used to predict tomorrow's sunshine hours for Basel. 

The notebooks in this folder are organized as follows:

- [generate_model.ipynb](generate_model_binary.ipynb) defines a neural network and trains it on Basel weather data from the light version of the weather prediction dataset.
