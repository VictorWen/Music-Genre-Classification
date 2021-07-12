import numpy as np
import pandas as pd
import datetime
import copy
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

def bootstrap_trials(base_model, n_trials, name, X, y, verbose=True):
  models = []
  # code for stable randomization
  mod = int(1E6) + 7
  code = (hash(name) ^ mod) % mod

  start_time = datetime.datetime.now()
  if verbose: print("Starting")
  for i in range(n_trials):
    trial_start = datetime.datetime.now()
    trial_X, trial_y = resample(X, y, random_state=code+i)
    model = copy.deepcopy(base_model)
    model.fit(trial_X, trial_y)
    if verbose: print("\rTrial:", i + 1, "Time:", str(datetime.datetime.now() - trial_start), "Total:", str(datetime.datetime.now() - start_time), end='')
    models.append(model)
  if verbose: print("\nElapsed Time:", str(datetime.datetime.now() - start_time))
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


def full_run(base_model, trials, name, folder, save_folder, verbose=True):
  print(name)
  train, test = load_dataset(folder)

  scaler = StandardScaler()
  labeler = LabelEncoder()
  X, y = prepare_dataset(scaler, labeler, train)
  X_test, y_test = prepare_dataset(scaler, labeler, test)

  models = bootstrap_trials(base_model, trials, name, X, y, verbose)
  accuracies, scores = evaluate_models(models, name, X_test, y_test, save_folder)
  lo_acc, mid_acc, hi_acc = get_CIs(accuracies)
  lo_f1, mid_f1, hi_f1 = get_CIs(scores)
  print("\tACCURACY:", f'{lo_acc*100:.3f} - {mid_acc*100:.3f} - {hi_acc*100:.3f}%')
  print("\tF1 SCORES:", f'{lo_f1*100:.3f} - {mid_f1*100:.3f} - {hi_f1*100:.3f}%')