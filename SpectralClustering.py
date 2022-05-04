# 在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。
# 同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。
# 最后一步的聚类方法则提供了两种，K-Means算法和discrete算法。
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, v_measure_score


# 执行1000次谱聚类，取最优结果返回
def RunSpectralClustering(X, y):
    y_best = y
    acc_best = 0
    n_clusters_best = 6
    gamma_best = 0.01
    for n_clusters in [6, 7, 8, 9]:
        for gamma in np.arange(0.01, 1.2, 0.05):
            sc = SpectralClustering(n_clusters=n_clusters, gamma=gamma, n_jobs=-1).fit(X)
            y_pred = sc.labels_
            acc = accuracy_score(y, y_pred)
            v_measure = v_measure_score(y, y_pred)
            if acc > acc_best:
                n_clusters_best = n_clusters
                gamma_best = gamma
                y_best = y_pred
                acc_best = acc
            print("Spectral Clustering " + (" gamma=" + str(gamma) + "; n_clusters=" + str(n_clusters)) + " Acc: " + str(acc) + "   v_measure:" + str(v_measure))
    print("Spectral Clustering Best:\n")
    print(y_best[0:100])
    print(n_clusters_best)
    print(gamma_best)
    print(acc_best)
    np.savetxt("y_best_pred_by_spectral_clustering", y_best)
    return y_best, n_clusters_best, gamma_best
