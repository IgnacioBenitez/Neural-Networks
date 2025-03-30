import numpy as np
import matplotlib.pyplot as plt

# Simple Perceptron Training with Multiple Inputs and Two Neurons
# Architecture:
# - 3 input neurons with fixed values
# - 2 neurons with a sigmoid activation function
# - 2 sets of 3 weights (w1, w2, w3 for each neuron) initialized randomly
# - Learning rate: 0.001
# - Single target output
# ---------------------------------------------------------------------------


# STEP 1 --> Define the functions for training the network
# Functions for forward and backward passes

def sigmoid(sop):
    return 1 / (1 + np.exp(-sop))

def error(predicted, target):
    return np.power(predicted - target, 2)

# Functions for backward pass
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

num_inputs = 3  # Number of inputs 
x = np.array([0.1, 0.3, 0.2]) 
w = np.random.rand(2, num_inputs)  # Initialize weights for the 2 neurons

target = np.array([0.3])  # Array for the output (multiple outputs are needed)
learning_rate = 0.001  # Learning rate hyperparameter

print("Initial weights: ", w)

for k in range(1000):
    iteration.append(k)
    
    # STEP 3: Forward Pass
    #sop = np.dot(x, w)
    mult = x * w
    sop = np.sum(mult)
    predicted = sigmoid(sop)
    err = error(predicted, target)
    
    errors.append(err)
    prediction.append(predicted)
    
    # STEP 4: Backward Pass
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop)
    g3 = deriv_sop_w(x)  # Gradient for the weights
    
    grad = g1 * g2 * g3  # g3 Gradient of the weights
    w = update_w(w, learning_rate, grad)  # Update all the weights

print("Final weights: ", w)

# Plot the evolution of the prediction
plt.plot(iteration, prediction)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.title("Evolution of the Prediction")
plt.show()

# Plot the evolution of the error
plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Evolution of the Error")
plt.show()
