import time
from scipy.io import loadmat
import numpy as np


def load():
    # 读取模拟信号特征矩阵，同时记录最大的shape
    # 读取.mat文件时的变量
    original_data = []
    file_path = "/Users/huyu/Library/Mobile Documents/com~apple~CloudDocs/PALM/模拟信号聚类/模拟信号（提取出来的特征）/"
    signal_i = 1
    work_i = 1
    # 记录最大shape的变量
    max_shape = np.array([0, 0])
    for train_i in range(1, 7):
        for file_i in range(1, 501):
            # 读取.mat文件
            file = loadmat(file_path + ("train" + str(train_i)) +
                           ("/signal" + str(signal_i) + "_work" + str(work_i) + "_secondDifference" + str(file_i)) +
                           ".mat")
            ccc = file.get("ccc")
            # 添加数据到list中，就不用多次读取了
            original_data.append(ccc)
            # 记录最大的shape以便后面对齐
            max_shape = [max(max_shape[0], ccc.shape[0]), max(max_shape[1], ccc.shape[1])]
        # 更新.mat文件信息
        if work_i == 2:
            signal_i += 1
            work_i = 1
        else:
            work_i += 1
    max_dim = max_shape[0] * max_shape[1]

    # 把特征全部转化为一维向量，并对齐特征长度
    data = np.array(np.zeros((3000, max_dim)))
    for i in range(len(original_data)):
        original_data[i] = original_data[i].reshape(1, -1)[0]
        if original_data[i].shape[0] < max_dim:
            data[i] = np.pad(original_data[i], (0, max_dim - original_data[i].shape[0]), "constant",
                             constant_values=(0, 0))

    print(data.shape)

    # 制作标签
    labels = np.array([np.ones((500,), dtype=int), 2 * np.ones((500,), dtype=int), 3 * np.ones((500,), dtype=int),
                       4 * np.ones((500,), dtype=int), 5 * np.ones((500,), dtype=int), 6 * np.ones((500,), dtype=int)])
    labels = labels.reshape((1, -1))[0]
    print(labels.shape)

    return data, labels


if __name__ == '__main__':
    print((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + " - START\n")
    X, y = load()
    print((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + " - DATA LOADING FINISHED\n")