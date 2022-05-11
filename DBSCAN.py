import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score



def RunDBSCAN(X, y):
    label_best = []
    ari_best = 0.0
    nmi_best = 0.0
    for e in np.arange(0.1, 0.9, 0.1):
        for ms in np.arange(10, 400, 10):
            clustering = DBSCAN(eps=e, min_samples=ms, n_jobs=-1).fit(X)
            pred_label = clustering.labels_
            n_clusters = len([i for i in set(pred_label) if i != -1])
            acc = accuracy_score(y, pred_label)
            nmi = normalized_mutual_info_score(y, pred_label)
            ari = adjusted_rand_score(y, pred_label)
            if nmi > nmi_best:
                label_best = pred_label
                nmi_best = nmi
            if ari > ari_best:
                label_best = pred_label
                ari_best = ari
            print("DBSCAN " + (str(e) + ";" + str(ms)) +
                  ": n_clusters:" + str(n_clusters) +
                  "     Acc: " + str(acc) +
                  "     nmi:" + str(nmi) +
                  "     ari:" + str(ari))
    print("DBSCAN Best: acc=" + str(acc) + " ; " + "ari=" + str(ari_best) + " ; " + "nmi=" + str(nmi_best))
    np.savetxt("y_best_pred_by_DBSCAN", label_best)
    return label_best
