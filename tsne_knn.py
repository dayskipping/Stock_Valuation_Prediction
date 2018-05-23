import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
plt.style.use('seaborn-colorblind')

# function for finding outliers
def is_outlier(col):
    mean = col.mean()
    sd = col.std()
    outliers = np.abs(col - mean) > sd*3
    return outliers

# Scatter plot function for mulit class data
def multi_class_scatter_plot(arr_1, arr_2, y, size, title):
    import matplotlib.colors as colors
    color_list = list(colors.cnames.keys())
    classes = np.unique(y)

    fig, ax = plt.subplots(figsize=size)
    for i in classes:
        color = np.random.choice(color_list)
        mask = y == i
        ax.scatter(arr_1[mask], arr_2[mask], c=color, label=f"{i}")

    ax.legend()
    plt.title(title)

# Plot silhouette scores to determine optimal number of n_clusters
def plot_silhouette(max_clusters, X):
    sil_scores = []
    x_values = []
    for i in range(2,max_clusters+1):
        x_values.append(i)
        clf = KMeans(n_clusters=i, max_iter=100, init='k-means++', n_init=10)
        labels = clf.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(x_values, sil_scores)
    plt.title('Silhouette Score')
    plt.xlabel('n_clusters')
    plt.ylabel('silhouette_score')

def K_means_clusters(X, n_clusters):
    n_clusters = n_clusters
    clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
    labels = clf.fit_predict(X)
    cluster_labels = labels.reshape(X.shape[0],1)
    X_embedded = np.append(X, cluster_labels, axis=1)
    return X_embedded, clf

def scaler(X):
    scaler = StandardScaler()
    X_transform = scaler.fit_transform(X)
    return X_transform

def make_stats_plots(stats_list, stats):
    fig, axes = plt.subplots(3,2, figsize=(16,16))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.bar(np.arange(len(stats_list[idx])), stats_list[idx])
        ax.set_xlabel('Clusters')
        ax.set_ylabel(stats[0].index[idx])
        ax.set_title(stats[0].index[idx])

def gridsearch(model, params, data):
    X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
    model = GridSearchCV(model(), param_grid=params, scoring='accuracy', n_jobs=-1, cv=5, verbose=1)
    model.fit(X_train, y_train)
    best_params = model.best_params_
    score = model.best_estimator_.score(X_test, y_test)

    return model
