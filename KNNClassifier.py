import pandas as pd
import numpy as np
import matplotlib as mtl
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

class KNNClassifier():

  def __init__(self):
    self.k = 0
    self.acu = 0
    self.fpath = 0
    self.train_data = []
    pass

  def train(self,train_path):
    data = pd.read_csv(train_path,header=None)
    array = pd.DataFrame(data).to_numpy()
    self.train_data = array
    train_data , validation_data = train_test_split(array, test_size=0.30, random_state=43)
    given_train_label = [a[0] for a in train_data]
    given_valid_label = [a[0] for a in validation_data]
    train_data = train_data[:,1:]
    validation_data = validation_data[:,1:]
    for k in range(3,10,2):
      knn_label = []
      for i in range(len(validation_data)):
        distance = []
        for j in range(len(train_data)):
          dis = euclidean_distance(validation_data[i],train_data[j])
          distance.append([dis,given_train_label[j]])
          distance.sort()
          distance = distance[0:k]
        label = assign_label(distance)
      knn_label.append(label)
      acc = accuracy_score(given_valid_label,knn_label)
      if acc > self.acu:
        self.acu = acc
        self.k = k

  def predict(self,test_path):
    test_data = pd.read_csv(test_path,header=None)
    train_array = self.train_data
    given_train_label =  train_array[:,0]
    train_array = train_array[:,1:]
    test_array = pd.DataFrame(test_data).to_numpy()
    k = self.k
    knn_label = []
    for i in range(len(test_array)):
      distance = []
      for j in range(len(train_array)):
        dis = euclidean_distance(test_array[i],train_array[j])
        distance.append([dis,given_train_label[j]])
        distance.sort()
        distance = distance[0:k]
      label = assign_label(distance)
      knn_label.append(label)
    return knn_label

  def euclidean_distance(data_point, training_data_point):
    a = np.array(data_point)
    b = np.array(training_data_point)
    c = a - b
    c = c*c
    c = math.sqrt(np.sum(c))
    return c

  def assign_label(label):
    lab_count = {}
    for a in label:
      if a[1] in lab_count:
        lab_count[a[1]] += 1
      else:
        lab_count[a[1]] = 1
    label_max = max(lab_count.items(), key = operator.itemgetter(1))[0]
    return label_max
  
  def find_accuracy(l1, l2):
    given_label = np.array(l1)
    knn_label = np.array(l2)
    label = given_label - knn_label
    per =  list(label).count(0)/len(given_label)
    return per
