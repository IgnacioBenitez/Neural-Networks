import numpy as np
import matplotlib.pyplot as plt

# Simple Perceptron Training with Multiple Inputs
# Architecture:
# - 10 input neurons (random values)
# - 1 neuron with a sigmoid activation function
# - 10 weights (w1, w2, ..., w10)
# - Learning rate: 0.01
# ---------------------------------------------------------------------------

# STEP 1 --> Define the training functions for the network
# Forward and backward functions

def sigmoid(sop):
    return 1 / (1 + np.exp(-sop))

def error(predicted, target):
    return np.power(predicted - target, 2)

# Backward functions
def deriv_error_predicted(predicted, target):
    return 2 * (predicted - target)

def deriv_activacion_sop(sop):
    sig = sigmoid(sop)
    return sig * (1 - sig)

def deriv_sop_w(x):
    return x

def update_w(w, learning_rate, grad):
    return w - learning_rate * grad

# STEP 2 --> Initialize values

errors = []
iteration = []
prediction = []

num_inputs = 10  # Number of inputs
x = np.random.rand(num_inputs) 
w = np.random.rand(num_inputs)  

target = 0.3
learning_rate = 0.01  # Learning rate hyperparameter

print("Initial weights: ", w)

for k in range(60000):
    iteration.append(k)
    
    # STEP 3: Forward Pass
    sop = np.dot(x, w) 
    predicted = sigmoid(sop)
    err = error(predicted, target)
    
    errors.append(err)
    prediction.append(predicted)
    
    # STEP 4: Backward Pass
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop)
    g3 = deriv_sop_w(x)  
    
    grad = g1 * g2 * g3  # Weight gradient
    w = update_w(w, learning_rate, grad)  # Update all weights

print("Final weights: ", w)

# Plot prediction evolution
plt.plot(iteration, prediction)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.title("Prediction Evolution")
plt.show()

# Plot error evolution
plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Error Evolution")
plt.show()
