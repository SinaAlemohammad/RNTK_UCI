import numpy as np
import sklearn
from sklearn.svm import SVC
from copy import deepcopy


def svm(K1, K2, y1, y2, C, c):
    n_val, n_train = K2.shape
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000, max_iter= 50000)
    clf.fit(K1, y1)
    z = clf.predict(K2)
    return 1.0 * np.sum(z == y2) / n_val
def normalizeData(X):
    n,L = X.shape
    max_norm = 0
    for i in range(n):
        temp = np.linalg.norm(X[i])
        if temp > max_norm:
            max_norm = temp
    X = X/max_norm
    return (X)
def svm2(K1, K2,y1,C, c):
    n_val, n_train = K2.shape
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000, max_iter= 50000)
    clf.fit(K1, y1)
    z = clf.predict(K2)
    return z


def Augdata(X,flip):
    n,L = X.shape
    x = deepcopy(X)
    if flip == 0:
       for i in range(n):
           x[i] = deepcopy(np.flip(X[i]))
    return x

