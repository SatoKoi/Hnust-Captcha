# -*- coding:utf-8 -*-
import sys
from PIL import Image
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def load_dataset():
    """训练集加载"""
    X = []
    y = []

    # 一共四十张图片
    for i in [1, 2, 3, "b", "c", "m", "n", "v", "x", "z"]:
        for j in range(4):
            path = "../dataset/{}{}.png".format(i, j)
            pix = np.array(Image.open(path))
            X.append(pix.reshape(9 * 20))  # 特征集, 特征长度为图片大小
            y.append(i)  # 分类集
    return np.array(X), np.array(y)


def split_letters(path):
    """分割单张图片"""
    pix = np.array(Image.open(path))
    start, step = 3, 9
    col_ranges = []
    for _ in range(4):
        col_ranges.append([start, start + step])
        start = start + step + 1
    letters = []
    for col_range in col_ranges:
        letter = pix[1:-1, col_range[0]: col_range[1]]
        letters.append(letter.reshape(9 * 20))
    return np.array(letters)


def search_super_params(X, y):
    """寻找合适的超参数"""
    params = [
        {
            "n_neighbors": [i for i in range(1, 10)],
            "weights": ["uniform"]
        },
        {
            "n_neighbors": [i for i in range(1, 10)],
            "weights": ['distance'],
            "p": [i for i in range(1, 5)]
        }
    ]
    knn_clf = KNeighborsClassifier()
    grid = GridSearchCV(knn_clf, params, verbose=2, n_jobs=2)
    grid.fit(X, y)
    print(grid.best_params_)        # get method = uniform, n_neighbors = 1


def accuracy_score(knn_clf):
    """准确度分析"""
    res = []
    prev = None
    for dir_path, dirs, _path in os.walk("../datasets"):
        for f in _path[125:]:
            if f.endswith((".png", ".jpg", ".jpeg")):
                pix = split_letters(os.path.join(dir_path, f))
                prev = np.vstack((prev, pix)) if prev is not None else pix
    with open("./y_test_data.txt", "r") as f:
        data = np.array([[code[0], code[1], code[2], code[3]] for code in f.readlines()]).reshape(125 * 4)
    return knn_clf.score(prev, data)


if __name__ == '__main__':
    X, y = load_dataset()
    # search_super_params(X, y)
    knn_clf = KNeighborsClassifier(n_neighbors=1)
    knn_clf.fit(X, y)
    # print(accuracy_score(knn_clf))
    prev = None
    for dir_path, dirs, _path in os.walk("../datasets"):
        for f in _path:
            if f.endswith((".png", ".jpg", ".jpeg")):
                pix = split_letters(os.path.join(dir_path, f))
                prev = np.vstack((prev, pix)) if prev is not None else pix
    print(knn_clf.predict(prev).reshape(-1, 4))
