import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score


def RunDBSCAN(X, y):
    label_best = []
    acc_best = -1.0
    for e in [1, 10, 100, 1000]:
        label = DBSCAN(eps=e, n_jobs=-1).fit_predict(X)
        # label = clustering.labels_
        acc = accuracy_score(y, label)
        if acc > acc_best:
            label_best = label
            acc_best = acc
        print("DBSCAN " + str(e) + " Scores: " + str(acc))
    print("DBSCAN Best Scores: " + str(acc_best))
    np.savetxt("y_best_pred", label_best)
