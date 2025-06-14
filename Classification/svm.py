# Load the important packages
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

# Load the dataset 
cancer = load_breast_cancer()
X = cancer.data[:, :2]  # Use only the first two features for visualization
Y = cancer.target

# Build the model 
svm = SVC(kernel='rbf', gamma=0.5, C=1.0)  # ✅ Fixed: 'rbc' → 'rbf', gamma should be float

# Train the model
svm.fit(X, Y)

# Plot the decision boundary 
disp = DecisionBoundaryDisplay.from_estimator(
    svm, X,
    response_method='predict',
    cmap=plt.cm.Spectral,
    alpha=0.8,
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1]
)

# Scatter plot of the data points
plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolors="k")
plt.title("SVM Decision Boundary")
plt.show()
