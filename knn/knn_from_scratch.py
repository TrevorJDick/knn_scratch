# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:01:45 2022

@author: TD
"""
import functools
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import interpolate
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


def knn_distance_matrix(X_true, X_score, distance_func):
    n_samples, n_features = X_true.shape
    n_score_samples, n_score_features = X_score.shape
    
    if n_features != n_score_features:
        raise ValueError(
            f'{n_features} != {n_score_features}, '
            'number of features dont match!'
        )
    
    # repeat copies of X_true along new axis for as many new score samples
    X_true = np.repeat(
        X_true.reshape((n_samples, n_features, 1)),
        n_score_samples,
        axis=2
    )
    # returns matrix where each column is distance from score row i to each sample
    d = distance_func(u=X_true, v=X_score.T)
    return d


def gen_nearest_matrices(distance_matrix):
    nearest_indices = np.argsort(distance_matrix, axis=0)
    nearest_distances = np.sort(distance_matrix, axis=0)
    return nearest_indices, nearest_distances


def get_k_nearest(X_score, y_score, nearest_indices, nearest_distances, k):
    """
    

    Parameters
    ----------
    X_score : TYPE
        DESCRIPTION.
    y_score : TYPE
        DESCRIPTION.
    nearest_indices : TYPE
        Indices of true arrays, ordered by increasing distance between
        score point and true point, for each score point.
    nearest_distances : TYPE
        Distances between score point and true point ordered by increasing
        distance for each score point.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    kn_indices : TYPE
        DESCRIPTION.
    kn_distances : TYPE
        DESCRIPTION.
    kn_pts : TYPE
        DESCRIPTION.
    kn_target : TYPE
        DESCRIPTION.

    """
    ### TODO unique indices for nearest k. could have many ties!
    kn_indices = nearest_indices[:k, :]
    kn_distances = nearest_distances[:k, :]
    kn_pts = X_score[kn_indices]
    # index loc on indeces 1 and 0, then 2 will give the k nearest pts
    # 0: k points, 1: n test points, 2: n features
    kn_target = y_score[kn_indices]
    return kn_indices, kn_distances, kn_pts, kn_target


def get_kn_pred(kn_target):
    # determine new target
    """
    1. for categorical get the most common class.
    2. for continuous any numerical centroid will work
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


def gen_X_test_lattice(X, N, x1_bounds=None, x2_bounds=None):
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    if x1_bounds is not None:
        x1_min, x1_max = x1_bounds
    if x2_bounds is not None:
        x2_min, x2_max = x2_bounds
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
        fpr, tpr, thresh = skm.roc_curve(y_test_ohc[:, i], y_pred_ohc[:, i])
        roc_auc = skm.roc_auc_score(y_test_ohc[:, i], y_pred_ohc[:, i])
        rocs.append((i, fpr, tpr, thresh, roc_auc))
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


def gen_roc_curves_multiclass(y_true, y_score):
    """
    Useful when doing many times with different split or for various k and
    generating mean roc curves withh std and auc.
    
    For multi classes problems we need to one hot encode. If binary the 
    one hot encoding will just create reshaped y arrays from 1d to 2d but with
    a single column.
    """
    label_binarizer = LabelBinarizer().fit(y_true)
    y_true_ohc = label_binarizer.transform(y_true)
    y_score_ohc = label_binarizer.transform(y_score)
    
    n_classes = y_true_ohc.shape[1]
    # keeping all 4 separate in case I have other designs
    fprs = []
    tprs = []
    threshes = []
    roc_aucs = []
    for i in range(n_classes):
        fpr, tpr, thresh = skm.roc_curve(
            y_true_ohc[:, i],
            y_score_ohc[:, i]
        )
        
        fprs.append(fpr)
        tprs.append(tpr)
        threshes.append(thresh)
        
        roc_auc = skm.roc_auc_score(
            y_true_ohc[:, i],
            y_score_ohc[:, i]
        )
        roc_aucs.append(roc_auc)
    return fprs, tprs, threshes, roc_aucs


def many_knn_runs(X, y, grid_params_basis=None, seed=73):
    if grid_params_basis is None:
        grid_params_basis = [(0.7, ), (5,)] # test_frac, k
    grid_params = list(itertools.product(*grid_params_basis))
    
    acc = []
    precision = []
    recall = []
    kfprs = []
    ktprs = []
    kthreshes = []
    raucs = []
    for test_frac, k in grid_params:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=seed
        )
        kn_indices, kn_distances, kn_pts, y_pred = run_knn(
            X_train,
            X_test, 
            y_train,
            y_test,
            k=k
        )
        
        # basically only care about metrics on test/validation set
        acc.append(skm.accuracy_score(y_test, y_pred))
        precision.append(skm.precision_score(y_test, y_pred, average=None))
        recall.append(skm.recall_score(y_test, y_pred, average=None))
        
        fprs, tprs, threshes, roc_aucs = gen_roc_curves_multiclass(
            y_test,
            y_pred
            )
        kfprs.append(fprs)
        ktprs.append(tprs)
        kthreshes.append(threshes)
        raucs.append(roc_aucs)
    
    acc = np.array(acc)
    precision = np.array(precision)
    recall = np.array(recall)
    
    ### below use to get centers and spreads for each class
    raucs = np.vstack(raucs) # shape (n_grid_params, n_classes)
    
    # for interpolate tpr and fpr
    n_classes = raucs.shape[1] # or len(kfprs[0]) would do
    interp_num = max(
        [
            max([i[j].shape[0] for i in kthreshes])
            for j in range(n_classes)
        ]
    )
    max_thresh = max(
        [
            max([np.max(i[j]) for i in kthreshes])
            for j in range(n_classes)
        ]
    )
    t = np.linspace(max_thresh, 0, num=interp_num) # new thresholds
    # interpolate so can stack to make matrix
    fprs_interp = [
        [
            interpolate.interp1d(h[j], fp[j], fill_value='extrapolate')(t)
            for fp, h in zip(kfprs, kthreshes)
        ] 
        for j in range(n_classes)
    ]
    fprs_interp = np.array(fprs_interp) # shape (n_classes, n_samples, n_threshes)
    tprs_interp = [
        [
            interpolate.interp1d(h[j], tp[j], fill_value='extrapolate')(t)
            for tp, h in zip(ktprs, kthreshes)
        ] 
        for j in range(n_classes)
    ]
    tprs_interp = np.array(tprs_interp)
    
    return (
        grid_params,
        (acc, precision, recall),
        (raucs, fprs_interp, tprs_interp)
    )


def plot_multiclass_auc_roc_curves_many(mean_fpr, mean_tpr, std_tpr,
                                        mean_auc, std_auc, n_std=1,
                                        grid_params=None):
    subtitle_str = ''
    if grid_params is not None:
        n_splits = len(set([i[0] for i in grid_params]))
        n_k = len(set([i[1] for i in grid_params]))
        subtitle_str = (
            f'\n{len(grid_params)} -- knn runs '
            f'(num train test splits:{n_splits}, num k:{n_k})'
        )
    
    colormap = sns.color_palette("viridis", as_cmap=True)
    colorspace = np.linspace(0, 1, mean_fpr.shape[1]) 
    plt.figure(figsize=(8, 8))
    for i in range(mean_fpr.shape[1]): # classes
        plt.plot(
            mean_fpr[:, i],
            mean_tpr[:, i],
            color=colormap(colorspace[i]),
            lw=1,
            label=(
                f'Class {i}, '
                'ROC curve '
                f'(area = {round(mean_auc[i], 4)}, '
                f'{n_std} std = {round(n_std * std_auc[i], 3)})'
            )
        )
        plt.fill_between(
            mean_fpr[:, i],
            mean_tpr[:, i] - n_std * std_tpr[:, i],
            mean_tpr[:, i] + n_std * std_tpr[:, i],
            color=colormap(colorspace[i]),
            alpha=0.2,
            label=f'$\pm$ {n_std} std. dev.',
        )
        del i
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
    plt.title(
        'Receiver operating characteristic for many knn runs'
        f'{subtitle_str}'
    )
    plt.legend(loc='lower right')
    plt.show()
    return


def plot_multiclass_metrics_many(grid_params, acc, precision, recall):
    plot_array = np.hstack(
        [
            np.concatenate([np.array(grid_params)] * precision.shape[1], axis=0),
            np.concatenate([acc] * precision.shape[1]).reshape(-1, 1),
            precision.ravel(order='F').reshape(-1, 1),
            recall.ravel(order='F').reshape(-1, 1),
            np.arange(precision.shape[1]).repeat(acc.shape[0]).reshape(-1, 1),
        ]
    )
    plot_array = pd.DataFrame(
        plot_array,
        columns=['test_frac', 'k', 'accuracy', 'precision', 'recall', 'class']
    )
    
    # all test metrics
    sns.relplot(
        x='k',
        y='accuracy',
        data=plot_array,
        hue='test_frac',
        col='class',
        col_wrap=2,
        kind='line',
        palette=sns.color_palette('plasma', as_cmap=True),
        lw=3,
    )
    plt.suptitle(
        'Test metrics for many knn runs',
        y=1.02
    )
    plt.show()
    
    sns.relplot(
        x='k',
        y='precision',
        data=plot_array,
        hue='test_frac',
        col='class',
        col_wrap=2,
        kind='line',
        palette=sns.color_palette('plasma', as_cmap=True),
        lw=3,
    )
    plt.suptitle(
        'Test metrics for many knn runs',
        y=1.02
    )
    plt.show()
    
    sns.relplot(
        x='k',
        y='recall',
        data=plot_array,
        hue='test_frac',
        col='class',
        col_wrap=2,
        kind='line',
        palette=sns.color_palette('plasma', as_cmap=True),
        lw=3,
    )
    plt.suptitle(
        'Test metrics for many knn runs',
        y=1.02
    )
    plt.show()
    return




'''
### get data
X, y = datasets.make_blobs(
    n_samples=100,
    centers=3,
    n_features=2,
    random_state=0
)


### for many splits and k see how well classifier performs
grid_params_basis = [
    [0.1, 0.2, 0.5, 0.8, 0.9], # test frac
    list(range(1, 30 + 1)) # k (be mindfull of k for given test fracs)
]

grid_params, test_scores, roc_auc_curves = many_knn_runs(
    X,
    y,
    grid_params_basis=grid_params_basis
)
acc, precision, recall = test_scores
raucs, fprs_interp, tprs_interp = roc_auc_curves
del test_scores, roc_auc_curves

# plot auc roc curves for many knn runs
mean_auc = np.mean(raucs, axis=0)
std_auc = np.std(raucs, axis=0)
mean_fpr = np.mean(fprs_interp, axis=1).T # shape (n_threhes, n_classes)
mean_tpr = np.mean(tprs_interp, axis=1).T
std_tpr = np.std(tprs_interp, axis=1).T
plot_multiclass_auc_roc_curves_many(mean_fpr, mean_tpr, std_tpr,
                                    mean_auc, std_auc, n_std=1,
                                    grid_params=grid_params)

# plot test metrics for many knn runs
plot_multiclass_metrics_many(grid_params, acc, precision, recall)

### get grid param that gives best acc for all classes]
# experimental
test_metrics = np.hstack(
    [
        np.array(grid_params),
        acc.reshape(-1, 1),
        precision, 
        recall
    ]
)

# first get only indices where acc is maximal
test_metrics_paredown = test_metrics[
    np.where(test_metrics[:, 2] == test_metrics[:, 2].max())[0]
]

# first we will sum the multiclass precision and recall columns
test_metrics_paredown = np.hstack(
    [
        test_metrics_paredown[:, :3],
        np.sum(test_metrics_paredown[:, 3: 6], axis=1).reshape(-1, 1),
        np.sum(test_metrics_paredown[:, 6: 9], axis=1).reshape(-1, 1),
    ]
)
# then maybe we care about recall the next
test_metrics_paredown = test_metrics_paredown[
    np.where(test_metrics_paredown[:, 4] == test_metrics_paredown[:, 4].max())[0]
]
# then precision
test_metrics_paredown = test_metrics_paredown[
    np.where(test_metrics_paredown[:, 3] == test_metrics_paredown[:, 3].max())[0]
]
# then small k (more precicesly we want the supremum- the least upper bound)
# 10% of smallest represented class
k_upper = 0.1 * np.unique(y, return_counts=True)[1].min()
test_metrics_paredown = test_metrics_paredown[
    np.where(test_metrics_paredown[:, 1] <= k_upper)[0]
]
# supremum
test_metrics_paredown = test_metrics_paredown[-1, :]
test_frac, k = test_metrics_paredown[:2]
k = int(k)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_frac,
    random_state=73
)

kn_indices, kn_distances, kn_pts, y_pred = run_knn(
    X_train,
    X_test, 
    y_train,
    y_test,
    k=k
)

print(
    f'test frac={test_frac}, k={k}:\n'
    f'{skm.accuracy_score(y_test, y_pred)} -- test accuracy\n'
    f'{skm.precision_score(y_test, y_pred, average=None)} -- test precision\n'
    f'{skm.recall_score(y_test, y_pred, average=None)} -- test recall\n'
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

N = 120
lattice = gen_X_test_lattice(X_train, N=N)
X_test_dense = np.column_stack((lattice[0].ravel(), lattice[1].ravel()))

kn_indices_dense, kn_distances_dense, kn_pts_dense, y_pred_dense = run_knn(
    X_train,
    X_test_dense, 
    y_train,
    y_test,
    k=1
)

plot_voroni_test(X_train, X_test_dense, y_train, y_pred_dense, vor, k=1)




### for avg of X show decision boundary, like having very small train frac
X_avg = np.vstack(
    [np.mean(X_test[np.where(y_test == i)], axis=0) for i in np.unique(y_test)]
)
y_avg = np.unique(y_test)

vor_avg = Voronoi(X_avg)
lattice_avg = gen_X_test_lattice(
    X_avg,
    N=N,
    x1_bounds=[-5, 5],
    x2_bounds=[-2, 7]
)
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


### knn on X_test_dense to compare against a knn on just avg pts
(
    kn_indices_extreme,
    kn_distances_extreme,
    kn_pts_extreme, 
    y_pred_extreme
) = run_knn(
    X_avg,
    X_test_dense, 
    y_avg,
    y_pred_dense,
    k=1
)

print(
    f'{skm.accuracy_score(y_pred_dense, y_pred_extreme)} -- test accuracy\n'
    f'{skm.precision_score(y_pred_dense, y_pred_extreme, average=None)} -- test precision\n'
    f'{skm.recall_score(y_pred_dense, y_pred_extreme, average=None)} -- test recall\n'
)

plot_test_pts_w_l2_circles(
    X_avg,
    X_test_dense,
    y_avg,
    kn_indices_extreme,
    kn_distances_extreme,
    kn_pts_extreme,
    y_pred_extreme,
    circles=False
)

rocs = roc_auc_plot(y_pred_dense, y_pred_extreme)
'''