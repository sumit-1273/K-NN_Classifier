import pandas as pd
import numpy as np
import matplotlib as mtl
import math
import operator
from sklearn.model_selection import train_test_split

def euclidean_distance(l1, l2):
  a = np.array(l1)
  b = np.array(l2)
  c = a - b
  c = c*c
  c = math.sqrt(sum(c))
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
  per =  label.count(0)/double(len(given_label))
  return per

data = pd.read_csv('train.csv',header=None)
array = pd.DataFrame(data).to_numpy()
train_data , validation_data = train_test_split(array, test_size=0.30, random_state=43)

given_train_label = [a[0] for a in train_data]
given_valid_label = [a[0] for a in validation_data]
knn_label = []

train_data = train_data[:,1:]
validation_data = validation_data[:,1:]

k = 3


for i in range(len(validation_data)):
  distance = []
  for i in  range(k):
    distance.append([math.inf,-1])
  for j in range(len(train_data)):
    dis = euclidean_distance(validation_data[i],train_data[j])
    distance.append([dis,given_train_label[j]])
    distance.sort()
    distance.pop()
  label = assign_label(distance)
  knn_label.append(label)
accuracy = find_accuracy(given_valid_label,knn_label)  
print(accuracy)
