import numpy as np
import matplotlib.pyplot as plt

# Neural Network with One Hidden Layer (2 Neurons)
# Architecture:
# - 3 input neurons (x1, x2, x3)
# - 1 hidden layer with 2 neurons
# - 1 output neuron with sigmoid activation
# - Learning rate: 0.01
# ---------------------------------------------------------------------------

# STEP 1 --> Define the training functions
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

num_inputs = 3  # Number of input neurons
x = np.array([0.1, 0.3, 0.2])  # Input values
w1_3 = np.random.rand(3)  # Weights for first hidden neuron
w2_3 = np.random.rand(3)  # Weights for second hidden neuron
w3_2 = np.random.rand(2)  # Weights for the output neuron

w3_2_old = w3_2.copy()

target = np.array([0.2])  # Target output
learning_rate = 0.01  # Learning rate hyperparameter

print("Initial weights: ", w1_3, w2_3, w3_2)

for k in range(10000):
    iteration.append(k)
    
    # STEP 3: Forward Pass
    sop1 = np.sum(w1_3 * x)  # Weighted sum for first hidden neuron
    sop2 = np.sum(w2_3 * x)  # Weighted sum for second hidden neuron

    # Activation of hidden layer
    sig1 = sigmoid(sop1)
    sig2 = sigmoid(sop2)

    sop3 = np.sum(w3_2 * np.array([sig1, sig2]))  # Weighted sum for output neuron

    predicted = sigmoid(sop3)  # Final prediction
    err = error(predicted, target)

    prediction.append(predicted)
    errors.append(err)

    # STEP 4: Backward Pass
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop3)
    g3 = np.zeros(w3_2.shape[0])  # Array with two zeros (for hidden layer)
        
    g3[0] = deriv_sop_w(sig1)
    g3[1] = deriv_sop_w(sig2)

    grad_hiden_output = g3 * g2 * g1  # Gradient for output layer

    # Update weights from hidden layer to output neuron
    w3_2 = update_w(w3_2, learning_rate, grad_hiden_output)

    # Update weights between input and hidden layer
    # First hidden neuron
    g3 = deriv_sop_w(w3_2_old[0]) 
    g4 = deriv_activacion_sop(sop1)
    g5 = deriv_sop_w(x)
    
    grad_hiden1_output = g5 * g4 * g3 * g2 * g1
    w1_3 = update_w(w1_3, learning_rate, grad_hiden1_output)

    # Second hidden neuron
    g3 = deriv_sop_w(w3_2_old[1]) 
    g4 = deriv_activacion_sop(sop2)
    g5 = deriv_sop_w(x)

    grad_hiden2_output = g5 * g4 * g3 * g2 * g1
    w2_3 = update_w(w2_3, learning_rate, grad_hiden2_output)

    w3_2_old = w3_2.copy()

print("Final weights: ", w1_3, w2_3, w3_2)

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
