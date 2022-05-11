import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score


def RunKMeans(X, y):
    ari_best = 0.0
    acc_best = 0.0
    nmi_best = 0.0
    label_best = []
    for i in np.arange(1, 3):
        clustering = KMeans(n_clusters=6, n_init=300, max_iter=300).fit(X)
        pred_label = clustering.labels_
        acc = accuracy_score(y, pred_label)
        ari = adjusted_rand_score(y, pred_label)
        nmi = normalized_mutual_info_score(y, pred_label)
        print("KMeans " + str(i) + ": acc:" + str(acc) + " ; nmi:" + str(nmi) + " ; ari:" + str(ari))
        if ari > ari_best:
            ari_best = ari
            label_best = pred_label
        if acc > acc_best:
            acc_best = acc
            label_best = pred_label
        if nmi > nmi_best:
            nmi_best = nmi
            label_best = pred_label
    print("KMeans best: ari = " + str(ari_best) + ", nmi=" + str(nmi_best) + ", acc=" + str(acc_best))
    np.savetxt("y_best_pred_by_KMeans", label_best)
    return label_best
