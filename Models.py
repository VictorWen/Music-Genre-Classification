import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import copy


def hierarchical_taxonomy(X, y, indicators, labels = None):
    if labels == None:
        label = range(0, len(indicators))
    conditions = []
    for indicator in indicators:
        conditions.append(np.isin(y, indicator))
    parent_y = np.select(conditions, labels)
    child_y = []
    child_X = []
    for condition in conditions:
        child_y.append(y[condition])
        child_X.append(X[condition])
    return parent_y, child_y, child_X


class HierarchicalClassifier:    
    
    def __init__(self, indicators, base_algorithm, labels = None, cv = True):
        self.roots = len(indicators)
        self.indicators = indicators
        self.labels = labels
        self.algorithms = []
        self.cv = cv
        self.base_algorithm = base_algorithm
        for i in range(0, self.roots):
            self.algorithms.append(copy.deepcopy(base_algorithm))
        self.parent_algorithm = copy.deepcopy(base_algorithm)
    
    def fit(self, X, y):
        parent_y, child_y, child_X = hierarchical_taxonomy(X, y, self.indicators, labels = self.labels)
        self.parent_algorithm.fit(X, parent_y)
        if self.cv:
            self.parent_algorithm = self.parent_algorithm.best_estimator_
        for i in range(self.roots):
            if len(self.indicators[i]) > 1:
                self.algorithms[i].fit(child_X[i], child_y[i])
                if self.cv:
                    self.algorithms[i] = self.algorithms[i].best_estimator_
            
    def parent_predict(self, X):
        return self.parent_algorithm.predict(X)
    
    def predict(self, X, parent_predict = None):
        if parent_predict is None:
            parent_predict = self.parent_predict(X)
        predict = []
        for i in range(len(X)):
            for j in range(self.roots):
                if parent_predict[i] == self.labels[j]:
                    if len(self.indicators[j]) > 1:
                        predict.append(self.algorithms[j].predict([X[i]]))
                    elif len(self.indicators[j]) == 1:
                        predict.append(self.indicators[j])
        return np.array(predict)


def apply_selection(X, selection, scale = True):
    selected = []
    if scale:
        ss = StandardScaler()
    for features in selection:
        if scale:
            selected.append(ss.fit_transform(X[features]))
        else:
            selected.append(X[features])
    return selected


class MultiCV:
    
    def __init__(self, base_cv, multi, refitted = True):
        self.models = []
        self.refitted = refitted
        for i in range(multi):
            self.models.append(copy.deepcopy(base_cv))
            
    def fit(self, Xs, y):
        for i, model in enumerate(self.models):
            model.fit(Xs[i], y)
            print(str(i + 1) + " " + str(datetime.datetime.now()))
            if self.refitted:
                self.models[i] = model.best_estimator_
            
    def predict(self, Xs):
        predictions = []
        for i, model in enumerate(self.models):
            predictions.append(model.predict(Xs[i]))
        return predictions