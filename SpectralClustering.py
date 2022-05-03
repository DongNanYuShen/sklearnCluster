# 在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。
# 同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。
# 最后一步的聚类方法则提供了两种，K-Means算法和discrete算法。
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score


# 执行1000次谱聚类，取最优结果返回
def spectral_clustering(X, y):
    n_clusters_array = np.array([6, 7, 8, 9])
    gamma_array = np.array([0.001, 0.01, 0.1, 1])
    y_best = y
    acc_best = 0
    n_clusters_best = 6
    gamma_best = 0.01
    for i in range(1000):
        n_clusters = np.random.choice(n_clusters_array)
        gamma = np.random.choice(gamma_array)
        y_pred = SpectralClustering(n_clusters=n_clusters, gamma=gamma).fit_predict(X)
        acc = accuracy_score(y, y_pred)
        if acc > acc_best:
            n_clusters_best = n_clusters
            gamma_best = gamma
            y_best = y_pred
        print("Spectral Clustering " + str(i) + " Scores: " + str(acc))
    print("Spectral Clustering Best:\n")
    print(y_best[0:100])
    print(n_clusters_best)
    print(gamma_best)
    return y_best, n_clusters_best, gamma_best
