# -*- coding: utf-8 -*-

__author__ = 'Xiaoqiang Liu'

import numpy as np
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree

def feature_extraction(cloud):
    """Split the feature from Lidar data
    cloud   np.arrary([43, n]),  include: is_ground, fpfh(33), label, number_of_return, return_num, intensity,
                                 normal_x, normal_y, normal_z, x, y, z
    return  the feature used split from cloud, include: fpfh, normal, height
    """
    
    #calculate the fpfh data
    fpfh = cloud[:, 1:34]
    
    # calculate the normal vector
    normal = cloud[:, 38:41]
    
    # calculate relative_height
    ground_xyz = cloud[cloud[:,0].astype(int)==1,-3:]
    kdt = KDTree(ground_xyz[:, 0:2], metric = 'euclidean')
    ind = kdt.query(cloud[:, -3:-1], k=1, return_distance = False)
    relative_height = cloud[:, -1] - ground_xyz[ind.flatten(), -1]
    relative_height = relative_height.reshape([cloud.shape[0], 1])
    
    # compose feature 
    feature = np.hstack((fpfh, normal))
    feature = np.hstack((feature, relative_height))
    return feature

def smooth_result(predict, ind):
    """
    A function to smooth the result of supervised learning using majority class of neighbourhood
    
    Arguments:
        predict : list or array of length N with the supervised learning result
        ind     :  A 2D numoy array of shape [N, K], K is the k-nn parameter
    """
    neigh_c = predict[ind]
    smooth_predict = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=neigh_c.astype('int'))
    
    return smooth_predict


def plot_confusion_matrix(confusion_matrix, class_name, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """Create a cofusion map from confusion matrix.

    confusion_matrix : A 2D numpy array of shape (N,N)
    class_name       : A list or array of length N with the name for classes
    
    Optional arguments:
        ax               : A matplotlib.axes.Axes instance to which the confusion matrix
                           is plotted. If not provided, use current axes or create a new one.
        cbar_kw          : A dictionary with arguments to `matplotlib.Figure.colorbar`.
        cbarlabel        : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the confusion matrix
    im = ax.imshow(np.log(confusion_matrix), **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 2, 4, 6, 8, 10], fraction=0.046, pad=0.04, **cbar_kw)
    cbar.ax.set_yticklabels(['$e^0$', '$e^2$', '$e^4$', '$e^6$', '$e^8$', '$e^{10}$'])
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(class_name, fontsize=7)
    ax.set_yticklabels(class_name, fontsize=7)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(confusion_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(confusion_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.ylabel("True label")
    plt.xlabel("Predict label")

    return im, cbar



def annotate_confusion(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.log(data).max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(np.log(data[i, j])) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    feature_extraction(0)