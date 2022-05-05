import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score

def RunMeanShift(X, y):
    label_best = []
    acc_best = -1.0
    v_measure_best = -1.0
    for i in np.arange(1, 2, 1):
        clustering = MeanShift(n_jobs=-1).fit(X)
        pred_label = clustering.labels_
        n_clusters = len(np.unique(pred_label))
        acc = accuracy_score(y, pred_label)
        v_measure = v_measure_score(y, pred_label)
        if acc > acc_best:
            label_best = pred_label
            acc_best = acc
        if v_measure > v_measure_best:
            label_best = pred_label
            v_measure_best = v_measure
        print("MeanShift " + str(i) + " : " +
              "     n_clusters:" + str(n_clusters) +
              "     Acc: " + str(acc) +
              "     v_measure:" + str(v_measure))
    print("MeanShift Best: " + str(acc_best) + " ; " + str(v_measure_best))
    np.savetxt("y_best_pred_by_MeanShift", label_best)
    return label_best
