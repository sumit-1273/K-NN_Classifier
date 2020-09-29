import pandas as pd
import numpy as np
import matplotlib as mtl
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import time

class KNNClassifier:

  def __init__(self):
    self.k = 0
    self.acu = 0
    self.path = ''
    self.train_data = []

  def euclidean_distance(self,data_point, training_data_point):
    a = np.array(data_point)
    b = np.array(training_data_point)
    c = a - b
    c = c*c
    c = np.sum(c)
    return c

  def eucl_distance(self,data_point, train):
    return np.sum((data_point-train)*(data_point-train),axis=1)

  def assign_label(self,label):
    lab_count = {}
    for a in label:
      if a[1] in lab_count:
        lab_count[a[1]] += 1
      else:
        lab_count[a[1]] = 1
    label_max = max(lab_count.items(), key = operator.itemgetter(1))[0]
    return label_max

  def voting_label(self,label):
    label_count = {}
    for a in label:
      if a in label_count:
        label_count[a] += 1
      else:
        label_count[a] = 1
    label_max = max(label_count.items(), key = operator.itemgetter(1))[0]
    return label_max
  
  def find_accuracy(self,l1, l2):
    given_label = np.array(l1)
    knn_label = np.array(l2)
    label = given_label - knn_label
    per =  list(label).count(0)/len(given_label)
    return per

  # Finding optimal value of K using training data
  def train(self,train_path):
    data = pd.read_csv(train_path,header=None)
    array = data.to_numpy()
    self.train_data = array
    train_data , validation_data = train_test_split(array, test_size=0.30, random_state=43)
    given_train_label = train_data[:,0]
    given_valid_label = validation_data[:,0]
    train_data = train_data[:,1:]
    validation_data = validation_data[:,1:]
    for k in range(3,4,2):
      knn_label = []
      for i in range(len(validation_data)):
        distance = self.eucl_distance(validation_data[i],train_data)
        labels = np.argsort(distance)
        k_labels = given_train_label[labels][:k]
        label = self.voting_label(k_labels)
        knn_label.append(label)
      acc = accuracy_score(given_valid_label,knn_label)
      if acc > self.acu:
        self.acu = acc
        self.k = k

  def predict(self,test_path):
    test_data = pd.read_csv(test_path,header=None)
    test_array  = test_data.to_numpy()
    k = self.k
    knn_label = []
    for i in range(len(test_array)):
      distance = self.eucl_distance(test_array[i],train_array)
      labels = np.argsort(distance)
      k_labels = given_train_label[labels][:k]
      label = self.voting_label(k_labels)
      knn_label.append(label)
    return knn_label
