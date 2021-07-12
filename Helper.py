import numpy as np
import pandas as pd
import datetime
import copy
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv

def load_dataset(folder):
  train = pd.read_csv(folder + "/train.csv", index_col = 0)
  test = pd.read_csv(folder + "/test.csv", index_col = 0)
  return train, test


def prepare_dataset(scaler, labeler, dataset):
  y = labeler.fit_transform(dataset.pop("genre"))
  X = scaler.fit_transform(dataset)
  return X, y

def get_SVC(gpu):
  return SVC() if not gpu else None

def bootstrap_trials(n_trials, name, params, X, y, verbose=True, gpu=False):
  models = []
  # code for stable randomization
  code = name.encode('utf-8')
  code = int.from_bytes(code, byteorder="big")
  code %= int(1E6) + 13

  start_time = datetime.datetime.now()
  if verbose: print("Starting")
  for i in range(n_trials):
    trial_start = datetime.datetime.now()
    trial_X, trial_y = resample(X, y, random_state=code+i)
    trial_SVC = get_SVC(gpu)
    trial_CV = GridSearchCV(trial_SVC, params, n_jobs=-1, cv=5)
    trial_CV.fit(trial_X, trial_y)
    print("\rTrial:", i + 1, "Time:", str(datetime.datetime.now() - trial_start), "Total:", str(datetime.datetime.now() - start_time), end='')
    models.append(trial_CV)
  print("\nElapsed Time:", str(datetime.datetime.now() - start_time))
  return models

def save_to_csv(name, accuracies, f1_scores, save_folder):
  with open(save_folder + "/" + name, "w+") as file:
    writer = csv.writer(file)
    writer.writerow(accuracies)
    writer.writerow(f1_scores)


def evaluate_models(models, name, test_X, test_y, save_folder):
  accuracies = []
  f1_scores = []
  for model in models:
      predict = model.predict(test_X)
      accuracy = accuracy_score(test_y, predict)
      f1 = f1_score(test_y, predict, average = 'macro')
      accuracies.append(accuracy)
      f1_scores.append(f1)
  save_to_csv(name + '.csv', accuracies, f1_scores, save_folder)
  return accuracies, f1_scores

def read_from_csv(name, save_folder):
  file_name = name + ".csv"
  with open(save_folder + "/" + file_name, 'r') as file:
    reader = csv.reader(file)
    rows = []
    for row in reader:
      rows.append(row)
  return rows[0], rows[1]


def get_CIs(scores, CONFIDENCE=95):
  alpha = 100 - CONFIDENCE
  low_percentile = alpha / 2.0
  high_percentile = 100 - low_percentile
  return np.percentile(scores, low_percentile), \
    np.median(scores), \
    np.percentile(scores, high_percentile)


def full_run(trials, name, params, folder, save_folder, verbose=True, gpu=False):
  print(name)
  train, test = load_dataset(folder)

  scaler = StandardScaler()
  labeler = LabelEncoder()
  X, y = prepare_dataset(scaler, labeler, train)
  X_test, y_test = prepare_dataset(scaler, labeler, test)

  models = bootstrap_trials(trials, name, params, X, y, verbose, gpu)
  accuracies, scores = evaluate_models(models, name, X_test, y_test, save_folder)
  lo_acc, mid_acc, hi_acc = get_CIs(accuracies)
  lo_f1, mid_f1, hi_f1 = get_CIs(scores)
  print("\tACCURACY:", f'{lo_acc*100:.3f} - {mid_acc*100:.3f} - {hi_acc*100:.3f}%')
  print("\tF1 SCORES:", f'{lo_f1*100:.3f} - {mid_f1*100:.3f} - {hi_f1*100:.3f}%')