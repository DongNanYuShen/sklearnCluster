# 在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。
# 同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。
# 最后一步的聚类方法则提供了两种，K-Means算法和discrete算法。

from sklearn.cluster import SpectralClustering
