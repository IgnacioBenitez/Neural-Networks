import numpy as np
import matplotlib.pyplot as plt

# Multi-Layer Perceptron Training with Two Hidden Layers
# Architecture:
# - 3 input neurons with fixed values
# - 1st hidden layer with 5 neurons, each using a sigmoid activation function
# - 5 sets of weights connecting the input layer to the 1st hidden layer (initialized randomly)
# - 2nd hidden layer with 3 neurons, each using a sigmoid activation function
# - 3 sets of weights connecting the 1st hidden layer to the 2nd hidden layer
# - 1 output neuron with a sigmoid activation function
# - 3 weights connecting the 2nd hidden layer to the output neuron
# - Learning rate: 0.01
# - Single target output
# ---------------------------------------------------------------------------


# STEP 1 --> Define the training functions for the network
# Functions for forward and backward

def sigmoid(sop):
    return 1 / (1 + np.exp(-sop))

def error(predicted, target):
    return np.power(predicted - target, 2)

# Functions for backward
def deriv_error_predicted(predicted, target):
    return 2 * (predicted - target)

def deriv_activation_sop(sop):
    sig = sigmoid(sop)
    return sig * (1 - sig)

def update_w(w, learning_rate, grad):
    return w - learning_rate * grad

# STEP 2 --> Initialize values

errors = []
iteration = []
prediction = []

num_inputs = 3  # Number of inputs
x = np.array([0.1, 0.4, 4.1]) 
target = np.array([0.2])  # For multiple outputs, an array is needed
learning_rate = 0.01  # Hyperparameter
#                              input layer, hidden layers, output layer
network_architecture = np.array([x.shape[0], 5, 3, 1])

w = []
w_temp = []

# Initialize weights
for layer in np.arange(network_architecture.shape[0] - 1):
    for neuron in np.arange(network_architecture[layer + 1]):
        w_temp.append(np.random.rand(network_architecture[layer]))  # Random initialization
    w.append(np.array(w_temp))
    w_temp = []

w_old = w

# Training the network
for k in range(1000):
    layer_idx = 0

    # Forward pass
    sop_hidden1 = np.matmul(w[layer_idx], x)  # Hidden layer 1
    sig_hidden1 = sigmoid(sop_hidden1)

    layer_idx += 1
    sop_hidden2 = np.matmul(w[layer_idx], sig_hidden1)  # Hidden layer 2
    sig_hidden2 = sigmoid(sop_hidden2)

    layer_idx += 1
    sop_output = np.dot(w[layer_idx], sig_hidden2)  # Output layer
    predicted = sigmoid(sop_output)
    err = error(predicted, target)

    errors.append(err)
    prediction.append(predicted)
    iteration.append(k)

    # Backward pass
    g1 = deriv_error_predicted(predicted, target)  # Error gradient
    g2 = deriv_activation_sop(sop_output)  # Activation gradient for the output layer
    grad_output = g1 * g2 * sig_hidden2  # Gradient for the output layer weights
    w[layer_idx] = update_w(w[layer_idx], learning_rate, grad_output)  # Update output layer weights

    # Backpropagate gradients
    g3 = deriv_activation_sop(sop_hidden2)  # Derivative of activation for hidden layer 2
    grad_hidden2 = g1 * g2 * g3  # Gradient for hidden layer 2

    layer_idx -= 1
    for neuron_idx in np.arange(w[layer_idx].shape[0]):
        grad_hidden1 = grad_hidden2 * sig_hidden1[neuron_idx] * g3[neuron_idx]  # Gradient for hidden layer 1
        w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], learning_rate, grad_hidden1)  # Update hidden layer 1 weights

# Plot the evolution of the prediction
plt.plot(iteration, prediction)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.title("Evolution of Prediction")
plt.show()

# Plot the evolution of the error
plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Evolution of Error")
plt.show()


# sklearn no se
# pythorch no se