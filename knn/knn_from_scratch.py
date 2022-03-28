# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:01:45 2022

@author: TD
"""
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import datasets
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def p_metric(u, v, p=2, axis=0):
    """
    Metrics (distance functions) induced from norms.
    """
    if p <= 0:
        raise ValueError(
            f'{p} -- lp-norms not well defined for p<=0'
        )
    elif (p > 0) and (p < 1):
        print(
            'WARNING: {p}-norm is not a norm.  Switching to an F-norm, which '
            'does induce a metric.'
        )
        norm = np.sum(np.abs(u - v) ** p, axis=axis) # note homogeneous part
    else:
        norm = np.sum(np.abs(u - v) ** p, axis=axis) ** (1 / p)
    return norm


def knn_distance_matrix(X_true, X_test, distance_func):
    n_samples, n_features = X_true.shape
    n_test_samples, n_test_features = X_test.shape
    
    if n_features != n_test_features:
        raise ValueError(
            f'{n_features} != {n_test_features}, number of features dont match!'
        )
    
    # repeat copies of X_true along new axis for as many new test samples
    X_true = np.repeat(
        X_true.reshape((n_samples, n_features, 1)),
        n_test_samples,
        axis=2
    )
    # returns matrix where each column is distance from test row i to each sample
    d = distance_func(u=X_true, v=X_test.T)
    return d


def gen_nearest_matrices(distance_matrix):
    nearest_indices = np.argsort(distance_matrix, axis=0)
    nearest_distances = np.sort(distance_matrix, axis=0)
    return nearest_indices, nearest_distances


def get_k_nearest(X, y, nearest_indices, nearest_distances, k):
    ### TODO unique indices for nearest k. could have many ties!
    kn_indices = nearest_indices[:k, :]
    kn_distances = nearest_distances[:k, :]
    kn_pts = X[kn_indices]
    # index loc on indeces 1 and 0, then 2 will give the k nearest pts
    # 0: k points, 1: n test points, 2: n features
    kn_target = y[kn_indices]
    return kn_indices, kn_distances, kn_pts, kn_target


def get_kn_pred(kn_target):
    # determine new target
    """
    1. for categorical get the most common class.
    2. for continuous any numberical centroid will work
    """
    # get most common class
    y_pred = []
    for i in range(kn_target.shape[1]):
        t, count = np.unique(kn_target[:, i], return_counts=True)
        t_mode = t[np.argsort(-count)][0]
        y_pred.append(t_mode)
        del t, count, t_mode, i
    y_pred = np.array(y_pred)
    return y_pred


def run_knn(X_train, X_test, y_train, y_test, k):
    metric_func = functools.partial(p_metric, p=2, axis=1)
    distance_matrix = knn_distance_matrix(X_train, X_test, metric_func)
    nearest_indices, nearest_distances = gen_nearest_matrices(distance_matrix)
    kn_indices, kn_distances, kn_pts, kn_target = get_k_nearest(
        X_train,
        y_train,
        nearest_indices,
        nearest_distances,
        k=k
    )
    y_pred = get_kn_pred(kn_target)
    return kn_indices, kn_distances, kn_pts, y_pred 


def plot_train_test(X_train, X_test, y_train, y_test):
    df_train = pd.DataFrame(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
        columns=['x1', 'x2'] + ['target']
    )
    df_test = pd.DataFrame(
        np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1),
        columns=['x1', 'x2'] + ['target']
    )

    colormap = sns.color_palette("viridis", as_cmap=True)
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        data=df_train,
        x='x1',
        y='x2',
        hue='target',
        palette=colormap,
        edgecolor='green',
        label='train',
    )
    sns.scatterplot(
        data=df_test,
        x='x1',
        y='x2',
        hue='target',
        palette=colormap,
        edgecolor='red',
        label='test',
    )
    plt.legend(loc='upper right')
    plt.show()
    return


def plot_test_pts_w_l2_circles(X_train, X_test, y_train, kn_indices,
                               kn_distances, kn_pts, y_pred, circles=True):
    df_train = pd.DataFrame(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
        columns=['x1', 'x2'] + ['target']
    )
    
    colormap = sns.color_palette("viridis", as_cmap=True)
    plt.figure(figsize=(12, 12))
    colorspace = np.linspace(0, 1, np.unique(y_train).shape[0]) # unique targets
    for i in range(kn_indices.shape[1]):
        nearest_X = kn_pts[:, i, :]
        color = colormap(colorspace[y_pred[i]])
        test_pt = X_test[i, :]
        if circles:
            max_kn_distances = np.max(kn_distances[:, i])
        
        if i == 0:
            plt.scatter(
                test_pt[0],
                test_pt[1],
                marker='X',
                color='black',
                s=200,
                label='test pt'
            )
        plt.scatter(
            test_pt[0],
            test_pt[1],
            marker='X',
            color=color,
            s=200,
        )
        if circles:
            plt.plot(
                test_pt[0] + (
                    max_kn_distances * np.cos(np.linspace(0, 2 * np.pi, 150))
                ),
                test_pt[1] + (
                    max_kn_distances * np.sin(np.linspace(0, 2 * np.pi, 150))
                ),
                color=color,
                lw=1,
            )
            plt.text(
                test_pt[0] + 1.1 * max_kn_distances * np.cos(-np.pi / 2),
                test_pt[1] + 1.1 * max_kn_distances * np.sin(-np.pi / 2),
                f'r={np.round(max_kn_distances, 3)}'
                )
        if i == 0:
            plt.scatter(
                nearest_X[:, 0],
                nearest_X[:, 1],
                marker='^',
                color='grey',
                s=200,
                label='k-nearest'
            )
        plt.scatter(
            nearest_X[:, 0],
            nearest_X[:, 1],
            marker='^',
            color=color,
            s=200,
        )
    sns.scatterplot(
        data=df_train,
        x='x1',
        y='x2',
        hue='target',
        palette=colormap,
        edgecolor='black',
    )
    plt.legend(loc='upper right')
    plt.show()
    return


def plot_voroni_train(X_train, y_train, vor):
    df_train = pd.DataFrame(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
        columns=['x1', 'x2'] + ['target']
    )
    
    colormap = sns.color_palette("viridis", as_cmap=True)
    plt.figure(figsize=(12, 12))
    axes = sns.scatterplot(
        data=df_train,
        x='x1',
        y='x2',
        hue='target',
        palette=colormap,
        edgecolor='black',
    )
    voronoi_plot_2d(
        vor,
        ax=axes,
        show_points=False,
        show_vertices=False,
        line_width=1.0,
    )
    plt.legend(loc='upper right')
    plt.show()
    return


def gen_X_test_lattice(X, N):
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    lattice = np.meshgrid(
        np.linspace(x1_min, x1_max, N),
        np.linspace(x2_min, x2_max, N)
    )
    return lattice


def plot_voroni_test(X_train, X_test, y_train, y_pred, vor, k):
    xlim = (-5, 5)
    ylim = (-2, 7)
    
    df_train = pd.DataFrame(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
        columns=['x1', 'x2'] + ['target']
    )
    
    colormap = sns.color_palette("viridis", as_cmap=True)
    plt.figure(figsize=(12, 12))
    colorspace = np.linspace(0, 1, np.unique(y_train).shape[0]) # unique targets
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker='X',
        color=colormap(colorspace[y_pred]),
        s=1,
    )
    axes = sns.scatterplot(
        data=df_train,
        x='x1',
        y='x2',
        hue='target',
        palette=colormap,
        edgecolor='black',
    )
    voronoi_plot_2d(
        vor,
        ax=axes,
        show_points=False,
        show_vertices=False,
        line_width=1.0,
    )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc='upper right')
    plt.title(f'{k} - nearest')
    plt.show()
    return


def roc_auc_plot(y_test, y_pred):
    # need to one hot encode test and pred to score
    label_binarizer = LabelBinarizer().fit(y_test)
    y_test_ohc = label_binarizer.transform(y_test)
    y_pred_ohc = label_binarizer.transform(y_pred)
    
    colormap = sns.color_palette("viridis", as_cmap=True)
    colorspace = np.linspace(0, 1, np.unique(y_test).shape[0]) 
    plt.figure(figsize=(8, 8))
    rocs = []
    for i in range(y_test_ohc.shape[1]):
        fpr, tpr, _ = skm.roc_curve(y_test_ohc[:, i], y_pred_ohc[:, i])
        roc_auc = skm.roc_auc_score(y_test_ohc[:, i], y_pred_ohc[:, i])
        rocs.append((i, fpr, tpr, roc_auc))
        plt.plot(
            fpr,
            tpr,
            color=colormap(colorspace[i]),
            lw=1,
            label=(
                f'Class {i}, ROC curve (area = {roc_auc})'
            )
        )
    plt.plot(
        [0, 1],
        [0, 1],
        color='black',
        lw=1, 
        linestyle='--',
        label='Chance',
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.show()
    return rocs



'''
### get data
X, y = datasets.make_blobs(
    n_samples=100,
    centers=3,
    n_features=2,
    random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.70,
    random_state=73
)

### do knn
kn_indices, kn_distances, kn_pts, y_pred = run_knn(
    X_train,
    X_test, 
    y_train,
    y_test,
    k=5
)

plot_test_pts_w_l2_circles(
    X_train,
    X_test,
    y_train,
    kn_indices,
    kn_distances,
    kn_pts,
    y_pred,
    circles=False
)

rocs = roc_auc_plot(y_test, y_pred)


### show Voroni connection
vor = Voronoi(X_train)

# plot_voroni_train(X_train, y_train, vor)

lattice = gen_X_test_lattice(X_train, N=100)
X_test_dense = np.column_stack((lattice[0].ravel(), lattice[1].ravel()))

kn_indices_dense, kn_distances_dense, kn_pts_dense, y_pred_dense = run_knn(
    X_train,
    X_test_dense, 
    y_train,
    y_test,
    k=1
)

plot_voroni_test(X_train, X_test_dense, y_train, y_pred_dense, vor, k=1)


### do ROC for many k, for fixed train test split, get mean, std roc_auc
# krocs = []
# for k in range(1, X_train.shape[0] + 1):
#     kn_indices, kn_distances, kn_pts, y_pred = run_knn(
#         X_train,
#         X_test, 
#         y_train,
#         y_test,
#         k=k
#     )
#     plot_test_pts_w_l2_circles(
#         X_train,
#         X_test,
#         y_train,
#         kn_indices,
#         kn_distances,
#         kn_pts,
#         y_pred,
#         circles=False
#     )
#     krocs.append(roc_auc_plot(y_test, y_pred))
#     del k, kn_indices, kn_distances, kn_pts, y_pred
### TODO krocs interpolate tpr and fpr arrays to same len to do aucroccurve avg,std


### for avg of X show decision boundary
X_avg = np.vstack(
    [np.mean(X_test[np.where(y_test == i)], axis=0) for i in np.unique(y_test)]
)
y_avg = np.unique(y_test)

vor_avg = Voronoi(X_avg)
lattice_avg = gen_X_test_lattice(X_avg, N=100)
X_avg_dense = np.column_stack((lattice_avg[0].ravel(), lattice_avg[1].ravel()))

(
    kn_indices_dense_avg,
    kn_distances_dense_avg,
    kn_pts_dense_avg, 
    y_pred_dense_avg
) = run_knn(
    X_train,
    X_avg, 
    y_train,
    y_avg,
    k=1
)

plot_voroni_test(X_avg, X_avg_dense, y_avg, y_pred_dense, vor_avg, k=1)


### show for very small train frac
X, y = datasets.make_blobs(
    n_samples=1000,
    centers=3,
    n_features=2,
    random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.99,
    random_state=73
)

### do knn
kn_indices, kn_distances, kn_pts, y_pred = run_knn(
    X_train,
    X_test, 
    y_train,
    y_test,
    k=5
)

vor = Voronoi(X_train)

lattice = gen_X_test_lattice(X_train, N=100)
X_test_dense = np.column_stack((lattice[0].ravel(), lattice[1].ravel()))

kn_indices_dense, kn_distances_dense, kn_pts_dense, y_pred_dense = run_knn(
    X_train,
    X_test_dense, 
    y_train,
    y_test,
    k=1
)

plot_voroni_test(X_train, X_test_dense, y_train, y_pred_dense, vor, k=1)
'''