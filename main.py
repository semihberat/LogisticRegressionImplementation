import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from LogisticModel import LogisticRegression

# Load the breast cancer dataset
df = load_breast_cancer() 
x,y = df["data"],df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42) # Split the data into training and testing sets
logModel = LogisticRegression() # Create an instance of the LogisticRegression class
logModel.fit(x_train,y_train) # Fit the model on the training data
predictions = logModel.predict(x_test) # Make predictions on the test data

# Calculate the accuracy of the predictions
def accuracy(predicted,test):
    return np.sum(predicted == test)/len(test)

# Print the accuracy
print(accuracy(predictions,y_test))
