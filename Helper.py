import numpy as np
import pandas as pd
import datetime
import copy
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import math
import pickle


def load_dataset(folder):
    train = pd.read_csv(folder + "/train.csv", index_col = 0)
    test = pd.read_csv(folder + "/test.csv", index_col = 0)
    return train, test


def separate_variables(dataset):
    y = dataset.pop("genre")
    X = dataset
    return X, y


def prepare_dataset(scaler, labeler, X, y):
    y = labeler.transform(y)
    X = scaler.transform(X)
    return X, y


def bootstrap_trials(base_model, n_trials, X, y, verbose=True, feature_table=None, random_start = None):
    models = []

    start_time = datetime.datetime.now()
    for i in range(n_trials):
        trial_start = datetime.datetime.now()
        trial_X, trial_y = resample(X, y, random_state=None if random_start is None else random_start + i)
        model = copy.deepcopy(base_model)
        if feature_table is not None:
            model.fit(trial_X, trial_y, selector__selected=feature_table[i])
        else:
            model.fit(trial_X, trial_y)
        if verbose: print("\rTrial:", i + 1, "Time:", str(datetime.datetime.now() - trial_start), "Total:", str(datetime.datetime.now() - start_time), end='')
        models.append(model)
    if verbose: print("\nElapsed Time:", str(datetime.datetime.now() - start_time))
    return models


def save_to_csv(name, rows, save_folder):
    with open(save_folder + "/" + name, "w+") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)


def parent_accuracy(test_y, predict, indicators):
    parent_y = np.zeros(test_y.shape)
    parent_predict = np.zeros(predict.shape)
    for i, indicator in enumerate(indicators):
        condition_y = np.isin(test_y, indicator)
        condition_predict = np.isin(predict, indicator)
        parent_y[condition_y] = i
        parent_predict[condition_predict] = i
    return accuracy_score(parent_y, parent_predict)


def evaluate_models(models, name, test_X, test_y, indicators, save_folder=None):
    accuracies = []
    f1_scores = []
    parent_scores = []
    for model in models:
        predict = model.predict(test_X)
        accuracy = accuracy_score(test_y, predict)
        f1 = f1_score(test_y, predict, average = 'macro')
        parent = parent_accuracy(test_y, predict, indicators)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        parent_scores.append(parent)
    if save_folder is not None: save_to_csv(name + '.csv', [accuracies, f1_scores, parent_scores], save_folder)
    return accuracies, f1_scores, parent_scores


def read_from_csv(name, save_folder):
    file_name = name + ".csv"
    with open(save_folder + "/" + file_name, 'r') as file:
        reader = csv.reader(file)
        rows = []
        for row in reader:
            rows.append(row)
    return rows[0], rows[1], rows[2]


def get_CIs(scores, CONFIDENCE=95):
    alpha = 100 - CONFIDENCE
    low_percentile = alpha / 2.0
    high_percentile = 100 - low_percentile
    return np.percentile(scores, low_percentile), \
        np.median(scores), \
        np.percentile(scores, high_percentile)
    
    
class ModelRunner:
    def __init__(self, base_model, trials, name, data_folder, indicators, save_folder, labeler=None, feature_table=None, random_start=None):
        self.base_model = base_model
        self.trials = trials
        self.last_saved = 0
        self.name = name
        self.data_folder = data_folder
        self.indicators = indicators
        self.save_folder = save_folder
        if labeler is None:
            labeler = LabelEncoder()
        self.labeler = labeler
        self.feature_table = feature_table
        self.random_start = random_start
     
        self.accuracies = []
        self.f1_scores = []
        self.parent_accuracies = []
    

    def run_models(self, verbose=True,save_rate=-1):
        print(self.name)
        
        train, test = load_dataset(self.data_folder)
        X, y = separate_variables(train)
        scaler = StandardScaler()
        scaler.fit(X)
        if self.labeler is None:
            self.labeler = LabelEncoder()
            self.labeler.fit(y)
        X, y = prepare_dataset(scaler, self.labeler, X, y)
        X_test, y_test = separate_variables(test)
        X_test, y_test = prepare_dataset(scaler, self.labeler, X_test, y_test)
        
        if save_rate < 0: save_rate = self.trials
        while self.last_saved < self.trials:
            print("CHECKPOINT START", self.last_saved, "/", self.trials)
            n_trials = min(self.trials-self.last_saved, save_rate)
            models = bootstrap_trials(self.base_model, n_trials, X, y, verbose=verbose, feature_table=self.feature_table, random_start=None if self.random_start is None else self.random_start + self.last_saved)
            a, f, p = evaluate_models(models, self.name, X_test, y_test, self.indicators)
            self.accuracies.extend(a)
            self.f1_scores.extend(f)
            self.parent_accuracies.extend(p)
            self.last_saved += n_trials
            with open(self.save_folder + '/' + self.name + ".pickle", 'wb+') as pickle_file:
                pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            print("CHECKPOINT END")

        self.get_results()


    def get_results(self):
        lo_acc, mid_acc, hi_acc = get_CIs(self.accuracies)
        lo_f1, mid_f1, hi_f1 = get_CIs(self.f1_scores)
        lo_p, mid_p, hi_p = get_CIs(self.parent_accuracies)
        print("ACCURACY:", f'{lo_acc*100:.3f} - {mid_acc*100:.3f} - {hi_acc*100:.3f}%')
        print("F1 SCORES:", f'{lo_f1*100:.3f} - {mid_f1*100:.3f} - {hi_f1*100:.3f}%')
        print("PARENT ACCURACIES:", f'{lo_p*100:.3f} - {mid_p*100:.3f} - {hi_p*100:.3f}%')


def load_model_runner(save_folder, name):
    with open(save_folder + '/' + name + ".pickle", 'rb') as pickle_file:
        return pickle.load(pickle_file)