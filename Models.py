import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import copy
import pandas as pd
from sklearn.base import BaseEstimator
import pymrmr

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
    
    def fit(self, X, y, **fit_params):
        parent_y, child_y, child_X = hierarchical_taxonomy(X, y, self.indicators, labels = self.labels)
        self.parent_algorithm.fit(X, parent_y, **fit_params)
        if self.cv:
            self.parent_algorithm = self.parent_algorithm.best_estimator_
        for i in range(self.roots):
            if len(self.indicators[i]) > 1:
                self.algorithms[i].fit(child_X[i], child_y[i], **fit_params)
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
                        predict.append(self.algorithms[j].predict(np.array([X[i]])))
                    elif len(self.indicators[j]) == 1:
                        predict.append(self.indicators[j])
        return np.array(predict)


class mRMRWrapper(BaseEstimator):
    def __init__(self, base_model, k=0.5):
        self.base_model = base_model
        self.k = k

    def fit(self, X, y):
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        lower = means - (stds * self.k)
        upper = means + (stds * self.k)
        below = X < lower
        above = X > upper
        discrete_features = np.select([below, above], [-1, 1], default = 0)
        n = X.shape[1]
        discrete_features = pd.DataFrame(discrete_features, columns = [str(i) for i in range(n)])
        self.selected_features = pymrmr.mRMR(discrete_features, 'MIQ', n)
        self.selected_features = [int(i) for i in self.selected_features]
        self.model = copy.deepcopy(self.base_model)
        self.model.fit(X, y, selector__selected=self.selected_features) # oddly assumes a pipline
        return self

    def predict(self, X):
        return self.model.predict(X)

    
class featureSelectionTransformer(BaseEstimator):
    def __init__(self, n_features=1):
        self.n_features = n_features
    def fit(self, X, y, selected):
        self.selected_features = selected[:self.n_features]
        return self

    def transform(self, X):
        selected_X = X[:, self.selected_features]
        return selected_X

class mRMRFeatureExtractor:
  def __init__(self, name, n_runs, random_start, save_folder, k = 0.5):
    self.count = 0
    self.name = name
    self.n_runs = n_runs
    self.random_start = random_start
    self.save_folder = save_folder
    self.k = k
    self.selected_features_table = []

  
  def fit(self, X, y):
    start_time = datetime.datetime.now()
    while self.count < self.n_runs:
      trial_X, trial_y = resample(X, y, random_state=self.random_start + self.count)
      
      means = trial_X.mean(axis=0)
      stds = trial_X.std(axis=0)
      
      lower = means - (stds * self.k)
      upper = means + (stds * self.k)
      
      below = trial_X < lower
      above = trial_X > upper
      
      discrete_features = np.select([below, above], [-1, 1], default = 0)
      n = trial_X.shape[1]
      
      discrete_features = pd.DataFrame(discrete_features, columns = [str(i) for i in range(n)])
      
      trial_start = datetime.datetime.now()
      selected_features = pymrmr.mRMR(discrete_features, 'MIQ', n)
      print("\rTrial:", self.count + 1, "Time:", str(datetime.datetime.now() - trial_start), "Total:", str(datetime.datetime.now() - start_time), end='')
      
      self.selected_features_table.append([int(i) for i in selected_features])
      
      self.count += 1
      with open(self.save_folder + '/' + self.name + ".pickle", 'wb+') as pickle_file:
        pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)