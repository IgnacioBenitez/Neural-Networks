import numpy as np
import matplotlib.pyplot as plt

# Multi-Layer Perceptron Training with One Hidden Layer
# Architecture:
# - 3 input neurons with fixed values
# - 1 hidden layer with 5 neurons, each using a sigmoid activation function
# - 5 sets of weights connecting the input layer to the hidden layer (initialized randomly)
# - 1 output neuron with a sigmoid activation function
# - 5 weights connecting the hidden layer to the output neuron
# - Learning rate: 0.01
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
x = np.array([0.1, 0.4, 4.1]) 
target = np.array([0.2])  # Array for multiple outputs
learning_rate = 0.01  # Hyperparameter
# Define network architecture: input layer, hidden layer with 5 neurons, output layer with 1 neuron
network_architecture = np.array([x.shape[0], 5, 1])

w = []
w_temp = []

# Initialize weights for each layer
for layer in np.arange(network_architecture.shape[0] - 1):
    for neuron in np.arange(network_architecture[layer + 1]):
        w_temp.append(np.random.rand(network_architecture[layer]))
    w.append(np.array(w_temp))
    w_temp = []

w_old = w
print("Initial weights: ", w)

# Training loop (forward pass and backward pass)
for k in range(1000):
    # ---------------- Forward pass ------------------
    sop_hidden1 = np.matmul(w[0], x)
    sig_hidden1 = sigmoid(sop_hidden1)
    sop_output = np.sum(w[2][0] * sig_hidden1)

    predicted = sigmoid(sop_output)
    err = error(predicted, target)

    errors.append(err)
    prediction.append(predicted)
    iteration.append(k)

    # ---------------- Backward pass ------------------
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop_output)
    g3 = deriv_sop_w(sig_hidden1)

    grad_hidden_output = g1 * g2 * g3
    w[1][0] = update_w(w[1][0], learning_rate, grad_hidden_output)

    g5 = deriv_sop_w(x)

    # Update weights between input layer and hidden layer
    for neuron_idx in np.arange(w[0].shape[0]):
        g3 = deriv_sop_w(w_old[1][0][neuron_idx])
        g4 = deriv_activacion_sop(sop_hidden1[neuron_idx])
        grad_hidden_input = g5 * g4 * g3 * g2 * g1
        w[0][neuron_idx] = update_w(w[0][neuron_idx], learning_rate, grad_hidden_input)
    
    w_old = w

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
