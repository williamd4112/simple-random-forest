import numpy as np
import numpy.matlib
import csv

from tqdm import *

from numpy import vstack,array
from numpy.random import rand

from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.cluster.vq import whiten

import cPickle as pickle

class Preprocessor(object):  
    def pca(self, X, k):
        pca = decomposition.PCA(n_components=k)
        pca.fit(X)
        self.model = pca      
        return pca.transform(X)
       
    def lda(self, X, T, d):
        clf = LinearDiscriminantAnalysis(n_components=d, solver='svd')
        clf.fit(X, T)
        self.model = clf
        return clf.transform(X)

    def normalize(self, X):
        return whiten(X), None

    def save(self, path):
        pickle.dump(self.model, open(path, "wb"), True) 

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

    def transform(self, X):
        return self.model.transform(X)

