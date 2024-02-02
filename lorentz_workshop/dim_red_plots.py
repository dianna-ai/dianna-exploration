from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import nptsne
import random

def pca(x):
    X_train                 = np.array(x)
    sc                      = StandardScaler()
    X_train_std             = sc.fit_transform(X_train)
    cov_mat                 = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs  = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    X_train_std[0].dot(w)

    X_train_pca = X_train_std.dot(w)

    return X_train_pca


def load_data(data_path, should_norm=False):
    data                    = np.load(os.path.join(data_path))  # pickle.load(open(data_path, 'rb'))
    data                    = dict(data)
    if should_norm and 'images' in data.keys():
        data['images']      = np.array(data['images']) / 255.
    return data

if __name__ == '__main__':


    MODEL_PATH              = './output/retrain_geometric_shapes_model.pt'
    IMG_DATA_TYPE           = 'colors'
    DIM_RED_TYPE            = 'pca' #'tsne'

    images                  = './data/test_colors.npz'
    heatmaps                = './data/test_colors_cams.npz',

    if IMG_DATA_TYPE == 'roundness':
        images              = images/255.


    if DIM_RED_TYPE == 'pca':
        embedding_img       = pca(images.reshape(images.shape[0], images.shape[1] * images.shape[2]))
        embedding_xai       = pca(heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1] * heatmaps.shape[2]))

    else:
        tsneobj             = nptsne.TextureTsne(False, 1000, 2, 20, 800, nptsne.KnnAlgorithm.Flann)
        random.seed(0)
        embedding_img       = tsneobj.fit_transform(images.reshape(images.shape[0], images.shape[1] * images.shape[2]))
        embedding_img       = embedding_img.reshape((int(embedding_img.shape[0] / 2), 2))
        random.seed(0)
        embedding_xai       = tsneobj.fit_transform(heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1] * heatmaps.shape[2]))
        embedding_xai       = embedding_xai.reshape((int(embedding_xai.shape[0] / 2), 2))

    fig, ax                 = plt.subplots(1, 2)
    viridis                 = cm.get_cmap('viridis', len(embedding_img))

    for i in range(len(embedding_img)):
        ax[0].scatter(embedding_img[i, 0], embedding_img[i, 1], c=viridis(i), alpha=0.5)
        ax[1].scatter(embedding_xai[i, 0], embedding_xai[i, 1], c=viridis(i), alpha=0.5)
    plt.show()