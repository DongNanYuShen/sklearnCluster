import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score


def RunDBSCAN(X, y):
    label_best = []
    acc_best = -1.0
    v_measure_best = -1.0
    for e in np.arange(0.001, 1.1, 0.05):
        for ms in np.arange(10, 400, 10):
            clustering = DBSCAN(eps=e, min_samples=ms, n_jobs=-1).fit(X)
            pred_label = clustering.labels_
            n_clusters = len([i for i in set(pred_label) if i != -1])
            acc = accuracy_score(y, pred_label)
            v_measure = v_measure_score(y, pred_label)
            if acc > acc_best:
                label_best = pred_label
                acc_best = acc
            if v_measure > v_measure_best:
                label_best = pred_label
                v_measure_best = v_measure
            print("DBSCAN " + (str(e) + ";" + str(ms)) +
                  ": n_clusters:" + str(n_clusters) +
                  "     Acc: " + str(acc) +
                  "     v_measure:" + str(v_measure))
    print("DBSCAN Best: " + str(acc_best) + " ; " + str(v_measure_best))
    np.savetxt("y_best_pred_by_DBSCAN", label_best)
