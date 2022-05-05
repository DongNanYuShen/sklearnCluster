import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score


def RunKMeans(X, y):
    acc_best = 0.0
    label_best = []
    for i in np.arange(1, 100):
        clustering = KMeans(n_clusters=6, n_init=20, max_iter=500).fit(X)
        pred_label = clustering.labels_
        acc = accuracy_score(y, pred_label)
        v_measure = v_measure_score(y, pred_label)
        print("KMeans " + str(i) + ": " + str(acc) + " ; " + str(v_measure))
        if acc > acc_best:
            acc_best = acc
            label_best = pred_label
    print("KMeans best acc = " + str(acc_best))
    np.savetxt("y_best_pred_by_KMeans", label_best)
    return label_best
