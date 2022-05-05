import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, v_measure_score, adjusted_rand_score



def RunDBSCAN(X, y):
    label_best = []
    ari_best = 0.0
    v_measure_best = 0.0
    for e in np.arange(0.1, 0.9, 0.1):
        for ms in np.arange(10, 400, 10):
            clustering = DBSCAN(eps=e, min_samples=ms, n_jobs=-1).fit(X)
            pred_label = clustering.labels_
            n_clusters = len([i for i in set(pred_label) if i != -1])
            acc = accuracy_score(y, pred_label)
            v_measure = v_measure_score(y, pred_label)
            ari = adjusted_rand_score(y, pred_label)
            if v_measure > v_measure_best:
                label_best = pred_label
                v_measure_best = v_measure
            if ari > ari_best:
                label_best = pred_label
                v_measure_best = v_measure
            print("DBSCAN " + (str(e) + ";" + str(ms)) +
                  ": n_clusters:" + str(n_clusters) +
                  "     Acc: " + str(acc) +
                  "     v_measure:" + str(v_measure) +
                  "     v_measure:" + str(ari))
    print("DBSCAN Best: v_measure=" + str(v_measure_best) + " ; " + "ari=" + str(ari_best))
    np.savetxt("y_best_pred_by_DBSCAN", label_best)
    return label_best
