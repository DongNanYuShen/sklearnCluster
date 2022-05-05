import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, v_measure_score, adjusted_rand_score


def RunKMeans(X, y):
    ari_best = 0.0
    label_best = []
    for i in np.arange(1, 10):
        clustering = KMeans(n_clusters=6, n_init=50, max_iter=600).fit(X)
        pred_label = clustering.labels_
        acc = accuracy_score(y, pred_label)
        ari = adjusted_rand_score(y, pred_label)
        v_measure = v_measure_score(y, pred_label)
        print("KMeans " + str(i) + ": " + str(acc) + " ; " + str(v_measure) + " ; " + str(ari))
        if ari > ari_best:
            ari_best = ari
            label_best = pred_label
    print("KMeans best ari = " + str(ari_best))
    np.savetxt("y_best_pred_by_KMeans", label_best)
    return label_best
