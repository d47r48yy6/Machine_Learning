# importing libraries
import numpy as np 
from collections import Counter

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1 ,point2):
    return np.sqrt(np.sum((np.array(point1)-np.array(point2))**2))
#Function for the Prediction using KNN
def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        distance = euclidean_distance(test_point, training_data[i])
        distances.append((distance, training_labels[i]))
    distances.sort(key=lambda x: x[0])  # Sort by distance
    k_nearest_labels = [label for _, label in distances[:k]]  # Get the labels of the k nearest neighbors
    return Counter(k_nearest_labels).most_common(1)[0][0]  # Return the most common label among the k nearest neighbors
# Traing data and labels

training_data = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
training_labels = np.array(['A', 'A', 'B', 'B', 'C'])
# Test pointEuclidean distance
test_point = np.array([4, 5])
# Number of neighbors to consider
k = 3

# Making a prediction
predicted_label = knn_predict(training_data, training_labels, test_point, k)
print(f"The predicted label for the test point {test_point} is: {predicted_label}")