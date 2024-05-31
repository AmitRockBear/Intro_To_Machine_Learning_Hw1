import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def calculate_euclidean_distance(vector1, vector2):
  return np.linalg.norm(vector1 - vector2)

# a
def predict_image_label(train_images, labels, query_image, k):
  unique_labels = np.unique(labels)
  labels_counter_dict = { value: 0 for value in unique_labels }
  distances = [calculate_euclidean_distance(image, query_image) for image in train_images]
  
  nearest_neighbors_sorted_idxs = np.argsort(distances)
  k_nearest_neighbors_sorted_idxs = nearest_neighbors_sorted_idxs[:k]
  
  for idx in k_nearest_neighbors_sorted_idxs:
    labels_counter_dict[labels[idx]]+=1

  return max(labels_counter_dict, key=labels_counter_dict.get)

# b
train_1000 = train[:1000]
train_labels_1000 = train_labels[:1000]

def calculate_label_prediction_accuracy_k(k):
  successful_predictions_counter = 0
  for index, test_image in enumerate(test):
    predicted_label = predict_image_label(train_1000, train_labels_1000, test_image, k)
    if predicted_label == test_labels[index]:
      successful_predictions_counter+=1
  return successful_predictions_counter / len(test)

accuracy = calculate_label_prediction_accuracy_k(10)
print("The accuracy of the prediction is:", accuracy)

# c
ks = list(range(1, 101))
accuracys = [calculate_label_prediction_accuracy_k(k) for k in ks]
plt.plot(ks, accuracys, label='Label Prediction Accuracy')
plt.xlabel('k')
plt.ylabel('Prediction Accuracy')
plt.title('Prediction Accuracy as a Function of k')
plt.legend()
plt.show()

# d
def calculate_label_prediction_accuracy_n(n):
  train_images_n = train[:n]
  train_labels_n = train_labels[:n]
  successful_predictions_counter = 0
  for index, test_image in enumerate(test):
    predicted_label = predict_image_label(train_images_n, train_labels_n, test_image, 1)
    if (predicted_label == test_labels[index]):
      successful_predictions_counter+=1
  return successful_predictions_counter / len(test)

ns = list(range(100, 5001, 100))
accuracys = [calculate_label_prediction_accuracy_n(n) for n in ns]
plt.plot(ns, accuracys, label='Label Prediction Accuracy')
plt.xlabel('n')
plt.ylabel('Prediction Accuracy')
plt.title('Prediction Accuracy as a Function of n')
plt.legend()
plt.show()

