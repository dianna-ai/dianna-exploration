## Binary MNIST dataset generation

The binary MNIST dataset is created by selecting two classes (0 and 1) from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It is generated for the exploration of XAI approaches in DIANNA project. 

In short, this dataset contains a train set of 12665 images depicting either a handwritten zero or one, and a test set of 2115 such images. Every image is stored as 28 x 28 greyscale pixels.

The whole pipeline of binary MNIST data handling and processing is described in the notebook [MNIST_exploration.ipynb](MNIST_exploration.ipynb).
