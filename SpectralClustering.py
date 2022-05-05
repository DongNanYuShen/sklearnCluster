# 在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。
# 同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。
# 最后一步的聚类方法则提供了两种，K-Means算法和discrete算法。
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score


# 执行sklearn谱聚类，取最优结果返回
def RunStanderSpectralClustering(X, y):
    y_best = y
    acc_best = 0.0
    ari_best = 0.0
    n_clusters_best = 6
    gamma_best = 0.01
    for n_clusters in [6, 7, 8, 9]:
        for gamma in np.arange(0.1, 0.7, 0.05):
            sc = SpectralClustering(n_clusters=n_clusters, gamma=gamma, n_jobs=-1).fit(X)
            y_pred = sc.labels_
            acc = accuracy_score(y, y_pred)
            ari = adjusted_rand_score(y, y_pred)
            if ari > ari_best:
                n_clusters_best = n_clusters
                gamma_best = gamma
                y_best = y_pred
                ari_best = ari
            print(
                "Spectral Clustering " + (" gamma=" + str(gamma) + "; n_clusters=" + str(n_clusters)) + " ari: " + str(
                    ari) + "   acc:" + str(acc))
    print("Spectral Clustering Best:" + " n=" +str(n_clusters_best) + " gamma=" + str(gamma_best) + " acc=" + str(acc_best) + " ari=" + str(ari_best))
    print(n_clusters_best)
    print(gamma_best)
    print(acc_best)
    np.savetxt("y_best_pred_by_spectral_clustering", y_best)
    return y_best


# 以下是从知乎拿下来的代码
def calculate_w_ij(a, b, sigma=1):
    w_ab = np.exp(-np.sum((a - b) ** 2) / (2 * sigma ** 2))
    return w_ab


# 计算邻接矩阵
def Construct_Matrix_W(data, k=5):
    rows = len(data)  # 取出数据行数
    W = np.zeros((rows, rows))  # 对矩阵进行初始化：初始化W为rows*rows的方阵
    for i in range(rows):  # 遍历行
        for j in range(rows):  # 遍历列
            if (i != j):  # 计算不重复点的距离
                W[i][j] = calculate_w_ij(data[i], data[j])  # 调用函数计算距离
        t = np.argsort(W[i, :])  # 对W中进行行排序，并提取对应索引
        for x in range(rows - k):  # 对W进行处理
            W[i][t[x]] = 0
    W = (W + W.T) / 2  # 主要是想处理可能存在的复数的虚部，都变为实数
    return W


def Calculate_Matrix_L_sym(W):  # 计算标准化的拉普拉斯矩阵
    degreeMatrix = np.sum(W, axis=1)  # 按照行对W矩阵进行求和
    L = np.diag(degreeMatrix) - W  # 计算对应的对角矩阵减去w
    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))  # D^(-1/2)
    L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix)  # D^(-1/2) L D^(-1/2)
    return L_sym


def normalization(matrix):  # 归一化
    sum = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))  # 求数组的正平方根
    nor_matrix = matrix / sum  # 求平均
    return nor_matrix


def RunMySpectralClustering(X, y):
    W = Construct_Matrix_W(X)  # 计算邻接矩阵
    L_sym = Calculate_Matrix_L_sym(W)  # 依据W计算标准化拉普拉斯矩阵
    lam, H = np.linalg.eig(L_sym)  # 特征值分解

    t = np.argsort(lam)  # 将lam中的元素进行排序，返回排序后的下标
    H = np.c_[H[:, t[0]], H[:, t[1]]]  # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
    H = normalization(H)  # 归一化处理

    model = KMeans(n_clusters=len(np.unique(y)))  # 新建KMeans模型
    model.fit(H)  # 训练
    labels = model.labels_  # 得到聚类后的每组数据对应的标签类型

    return labels
