## LeafSnap30 dataset <img width="30" alt="LeafSnap30 Logo" src="https://user-images.githubusercontent.com/3244249/150962983-8d8cb8f3-e8ea-48cd-a624-ef16768e1897.png"> generation

The original LeafSnap dataset has been created to facilite the automatic classification of tree species based on the images of their leaves. It has been downloaded from [kaggle.com](https://www.kaggle.com/xhlulu/leafsnap-dataset) as it was not avaialbe at the original location [leafsnap.com](http://leafsnap.com/dataset/) at the time. There are 30 866 (~31k) color images of different sizes. The dataset covers all 185 tree species from the Northeastern United States. The original images of leaves taken from two different sources:

    "Lab" images, consisting of high-quality images taken of pressed leaves, from the Smithsonian collection.
    "Field" images, consisting of "typical" images taken by mobile devices (iPhones mostly) in outdoor environments.

For the purpose of DIANNA a subset of 30 species has been selected, the LeafSnap30 dataset. The 30 most populous in the number of images per species have been chosen resulting in 7395 images divided in 5917 training, 739 validation samples and 739 test samples.

This folder contains 2 notebooks: Data_exploration and Image_cropping.
- [Data_exploration](Data_exploration.ipynb)
The purpose of this notebook is to select a subset of the most populous 30 species of lab and field images. Already a [dataset of 30 classes](https://github.com/NLeSC/XAI/blob/master/Software/LeafSnapDemo/Data_preparation_30subset.ipynb) have been selected before, where for the lab images have been cropped semi-manually using IrfanView to remove the riles and color calibration image parts. But 2/3 of that dataset has been selected randomly, not according to the number of images in that class.

This notebook is used to explore the original dataset and find out the most polpulous 30 classes and see which have not been included yet in the previous 30-class dataset.

- [Image_cropping](Image_cropping.ipynb)
The purpose of this notebook is the croping of the lab images of some species in the LeafSnap30 dataset.

- [Train_test_split.ipynb](Train_test_split.ipynb)
The purpose of this notebook is to split the data in a train, test, and validation set. This is done by creating a new folder with symbolic links to the original images.
