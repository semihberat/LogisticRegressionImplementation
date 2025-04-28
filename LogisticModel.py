import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))

class LogisticRegression():
    def __init__(self,alpha= 0.001, iter = 1000):
        self.alpha = alpha
        self.iter = iter
        self.weights = None
        self.bias = 0
        
    def fit(self,x,y):
        n_samples,n_features = x.shape  # Get the number of samples and features in the input data
        self.weights = np.zeros(n_features) # Initialize the weights to zeros
        
        for _ in range(self.iter): # Perform gradient descent for the specified number of iterations
            
            linear_calc = np.dot(x,self.weights) + self.bias # Calculate the linear equation
            predictions = sigmoid(linear_calc) # Apply the sigmoid function to the linear equation to obtain probabilities
            
            # Calculate the gradients
            dw = (1/n_samples)*np.dot(x.T,(predictions-y))
            db = (1/n_samples)*np.sum(predictions-y)
            
            # Update the weights and bias using the gradients and learning rate
            self.weights = self.weights - self.alpha*dw
            self.bias = self.bias - self.alpha*db
            
    def predict(self,x):
        linear_calc = np.dot(x,self.weights) + self.bias # Calculate the linear equation
        predictions = sigmoid(linear_calc) # Apply the sigmoid function to the linear equation to obtain probabilities
        results = [0 if y < 0.5 else 1 for y in predictions] # Convert probabilities to binary predictions
        return results
        
