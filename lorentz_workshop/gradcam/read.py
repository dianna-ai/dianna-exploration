from torch.nn.functional import mse_loss, softmax
from scipy.stats import wasserstein_distance
from scipy.stats import (pearsonr, spearmanr, pointbiserialr, kendalltau, weightedtau,
                         somersd, multiscale_graphcorr)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pingouin import distance_corr

import matplotlib.cm as cm
np.random.seed(19680801)

def mgc_plot(x, y, sim_name, mgc_dict=None, only_viz=False,
             only_mgc=False):

    """Plot sim and MGC-plot"""
    if not only_mgc:
        # simulation
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_title(sim_name + " Simulation", fontsize=20)
        ax.scatter(x, y)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.axis('equal')
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        plt.show()

    if not only_viz:
        # local correlation map
        plt.figure(figsize=(8,8))
        ax = plt.gca()
        mgc_map = mgc_dict["mgc_map"]

        # draw heatmap
        ax.set_title("Local Correlation Map", fontsize=20)
        im = ax.imshow(mgc_map, cmap='YlGnBu')

        # colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")
        ax.invert_yaxis()

        # turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        # optimal scale
        opt_scale = mgc_dict["opt_scale"]
        ax.scatter(opt_scale[0], opt_scale[1], marker='X', s=200, color='red')

        # other formatting
        ax.tick_params(bottom="off", left="off")
        ax.set_xlabel('#Neighbors for X', fontsize=15)
        ax.set_ylabel('#Neighbors for Y', fontsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.show()

def minmax(values):
    return (values -values.min()) / (values.max() - values.min())

def plotting():
    sns.lineplot(x=x_values[1:], y=changes)
    plt.show()

    sns.lineplot(x=x_values[1:], y=changes_original)
    plt.xlabel("Delta continouity")
    plt.ylabel("Wasserstein distance")
    plt.show()

def correlation_metric(x_values, changes_original):
    corr_list = (pearsonr, spearmanr, pointbiserialr, kendalltau, weightedtau,
                 somersd, multiscale_graphcorr, distance_corr)

    for corr_metric in corr_list:
        res_metric = corr_metric(x_values, changes_original)
        print(corr_metric.__name__, end=", ")
        if corr_metric == multiscale_graphcorr:
            print(f"statistic: {res_metric.statistic}, p-value: {res_metric.pvalue}")
            mgc_dict = res_metric.mgc_dict
            mgc_plot(x_values, changes_original,"Unknown", mgc_dict=mgc_dict, only_mgc=True)
            # print(f"correlaton: {res_metric}")
        elif corr_metric == distance_corr:
            print(f"correlation: {res_metric[0]}, p-value: {res_metric[1]}")
        else:
            print(f"statistic: {res_metric.statistic}, p-value: {res_metric.pvalue}")


if __name__ == '__main__':

    path = '../../example_data/dataset_preparation/geometric_shapes/test_roundedness.npz'
    path1 = '../dataset_roundedness.npz'
    pictures = np.load(path)

    """
    print(pictures.files, len(pictures['images']))

    plt.imshow(pictures['images'][0])
    plt.show()

    plt.imshow(pictures['images'][10])
    plt.show()

    plt.imshow(pictures['images'][20])
    plt.show()

    plt.imshow(pictures['images'][30])
    plt.show()

    plt.imshow(pictures['images'][50])
    plt.show()

    plt.imshow(pictures['images'][70])
    plt.show()

    plt.imshow(pictures['images'][80])
    plt.show()

    plt.imshow(pictures['images'][99])
    plt.show()
    """

    output = np.load('./outheatmap.npz', allow_pickle=True)

    data_path = str(output['data_path'])
    print(data_path)

    if 'color' in data_path.split('/')[-1]:
        x_values = output['color']
    elif 'rotation' in data_path.split('/')[-1]:
        x_values = output['rotation']
    elif 'roundedness' in data_path.split('/')[-1]:
        x_values = output['roundedness']


    heatmaps = torch.tensor(output['heatmaps'])

    metric = wasserstein_distance
    transform = softmax

    changes = np.zeros(len(heatmaps)-1)
    changes_original = np.zeros(len(output['heatmaps'])-1)

    with torch.no_grad():
        for i in range(len(output['heatmaps'])-1):
            adj_change = metric(
                transform(heatmaps[i+1, ...].flatten(), dim=0),
                transform(heatmaps[i, ...].flatten(), dim=0))
            changes[i] = adj_change
            changes_original[i] = metric(transform(heatmaps[i+1, ...].flatten(), dim=0),
                                         transform(heatmaps[0, ...].flatten(), dim=0))

    plotting()
    correlation_metric(x_values[1:][~np.isnan(changes_original)], changes_original[~np.isnan(changes_original)])
    kendall = kendalltau(x_values[1:][~np.isnan(changes_original)], changes_original[~np.isnan(changes_original)])
    print(kendall.pvalue, kendall.statistic)