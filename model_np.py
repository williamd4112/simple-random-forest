import logging
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

import cPickle as pickle

class Model(object):
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"), True) 

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))
    
    def test(self, X):
        return self.model.predict(X)

    def eval(self, X, T):
        y = self.test(X).astype(np.int32)
        t = T.astype(np.int32)
        return float(np.equal(y, t).sum()) / len(t)

class RandomForest(Model):
    def train(self, X, T, param):
        logging.info('X = [%d, %d], T = [1]' % (X.shape[0], X.shape[1]))
        for c in range(int(np.min(T)), int(np.max(T) + 1)):
            logging.info('#Class %d = %d' % (c, (T == c).sum()))
        self.model = BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=int(param[1])),
                                        n_estimators=int(param[0]),
                                        max_samples=float(param[2]),
                                        bootstrap=True)
        self.model.fit(X, T)

