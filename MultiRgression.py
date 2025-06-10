#Importing necessary libraries in the code 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing      # California housing dataset which was imported from sklearn.datasets

#Load the california housing dataset
housing = fetch_california_housing()
df =pd.DataFrame(housing.data, columns=housing.feature_names)

#Add the tareget variable to the dataframe
df['PRICE'] =housing.target

#Display the few rows of the dataframe
print(df.head())

#check the missing values 
print(df.isnull().sum())
#Define the feature (indenpendent variable) and target (dependent variable)
X = df[['MedInc']]  #Median income (similar correlation to the target variable)
y = df['PRICE']     #House price
 
#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size : {X_test.shape[0]}")
#create a simple linear regression model
model =LinearRegression()
#Train the model on the trainig data
model.fit(X_train, y_train)
#Print the intercept and coefficients 
print(f"Intercept: {model.intercept_}")
print(f"Coefficient :{model.coef_}")
#Predict the house prices for the test set
y_pred = model.predict(X_test)
#Display the few predicted values
predictions =pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(predictions.head())
#Plot the actual data points (Red dots are the data points)
plt.scatter(X_test, y_test, color='red', label='Actual') 
#Plot the regression line (Blue line is the regression line)
plt.plot(X_test, y_pred, color='blue', label='Regression Line')
#Add labels and the title 
plt.xlabel('Median income (MedInc)')
plt.ylabel('House Price ($100,000)')
plt.title('Simple Linear Regression : House Price vs Median Income')
plt.legend()  # Add the legend to the plot
plt.show()
#Calculate the mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
#Calculate the R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")   
 



