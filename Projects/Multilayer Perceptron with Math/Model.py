import numpy as np


#Architecture defination
def network_architecture(X, Y):
    n_x = X.shape[1]  # number of input features
    n_h = 2            # hidden layer neurons
    n_y = Y.shape[1]   # output neurons
    return (n_x, n_h, n_y)



#Parameters initialization
def initialization(n_x, n_h, n_y, W_factor=0.01, b_factor=0):
    # 2x2 weight matrix for hidden layer, random numbers from standard normal distribution
    W1 = np.random.randn(n_x, n_h) * W_factor

    # 1x2 bias vector for hidden layer, initialized to zeros
    b1 = np.zeros((1, n_h)) + b_factor

    # 2x1 weight matrix for output layer, random numbers from standard normal distribution
    W2 = np.random.randn(n_h, n_y) * W_factor

    # 1x1 bias vector for output layer, initialized to zeros
    b2 = np.zeros((1, n_y)) + b_factor

    # Store parameters in a dictionary for easy access
    parameters = { 
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters



#Activation functions

def relu_forward(Z):
    # ReLU activation: replaces negative values in Z with 0, keeps positive values unchanged
    return np.maximum(0, Z)


def sigmoid_forward(Z):
    # Sigmoid activation: maps any real number Z to a value between 0 and 1
    return 1 /(1+np.exp(-Z))



#Forward Propigation in Code
def forward_propagation(X, parameters):
    #retrieve parameters for readability
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    #hidden layer affine combination
    Z1 = X @ W1 + b1
    
    #hidden layer activation (ReLU)
    A1 = relu_forward(Z1)
    
    #output layer affine combination
    Z2 = A1 @ W2 + b2
    
    #output layer activation (Sigmoid)
    Y_hat = sigmoid_forward(Z2)
    
    #store intermediate values in cache for backpropagation
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "Y_hat": Y_hat
    }
    
    return Y_hat, cache



#Loss Function
def loss_function(Y_hat,Y):
    #will accept the the predicted and the true label and return the error
    m = Y.shape[0]
    loss = 1/m * np.sum(-(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat)))
    return loss

#Backward Propigation in Code
def backward_propagation(parameters, cache, X, Y):
    #retrieve parameters
    W2 = parameters["W2"]

    #retrieve cache
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Y_hat = cache["Y_hat"]
    m = X.shape[0]

    #gradients of the output layer
    dZ2 = Y_hat - Y
    dW2 = 1/m * (A1.T @ dZ2)
    db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True)

    #gradients of the hidden layer
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = 1/m * (X.T @ dZ1)
    db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True)

    #store the gradients in a dictionary
    gradients = {
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }
    return gradients

#Updating the Parameters: Gradient Descent in Code
def gradient_descent(parameters, gradients, learning_rate=0.001):
    #retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #retrieve gradients
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    #update parameters using gradient descent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    #store updated parameters in a dictionary
    parameters = { 
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

#The model class
class Model:
    #this method will be used to train the model
    def fit(self, X, Y, epoch, learning_rate=0.01, W_factor=0.01, b_factor=0, print_loss=False, print_parameters=False):
        #check to see if the shape of X and Y match before anything
        assert X.shape[0] == Y.shape[0]
        #define the network architecture
        n_x, n_h, n_y = network_architecture(X, Y)

        #initialize the network parameters (weights and biases)
        parameters = initialization(n_x, n_h, n_y, W_factor=0.01, b_factor=0)

        #training loop
        for i in range(epoch):
            # Forward propagation: compute predictions and cache intermediate values
            Y_hat, cache = forward_propagation(X, parameters)

            #compute the loss for this iteration
            loss = loss_function(Y_hat, Y)

            #backward propagation: compute gradients w.r.t. parameters
            gradients = backward_propagation(parameters, cache, X, Y)

            #update parameters using gradient descent
            parameters = gradient_descent(parameters, gradients, learning_rate)

            #print loss to see how the model improves between iterations
            if print_loss:
                print(f"The loss of iteration number {i} is {loss}")

        #store the trained parameters as an instance variable to be used in the prediction method
        self.model_parameters = parameters

        #print parameters to see the model parameters
        if print_parameters:
            print(f"The model parameters are {self.model_parameters}")

    def predict(self, X):
        #this method accepts the input data and returns the predictions using the trained model parameters
        Y_hat, _ = forward_propagation(X, self.model_parameters)
        #convert probabilities to class labels: if Y_hat >= 0.5 → class 1, else class 0
        return (Y_hat >= 0.5).astype(int)
