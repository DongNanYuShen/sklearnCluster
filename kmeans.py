import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score


def RunKMeans(X, y):
    for i in np.arange(1, 10):
        clustering = KMeans(n_clusters=6, n_init=100,  max_iter=500).fit(X)
        pred_label = clustering.labels_
        acc = accuracy_score(y, pred_label)
        v_measure = v_measure_score(y, pred_label)
        print("KMeans " + str(i) + ": " + str(acc) + " ; " + str(v_measure))
