import numpy as np
import matplotlib.pyplot as plt

# Simple Neural Network Training with a Hidden Layer
# Architecture:
# - 3 input neurons with fixed values
# - 1 hidden layer with 5 neurons, each using a sigmoid activation function
# - 5 sets of 3 weights (one per hidden neuron), initialized randomly
# - 1 output neuron with a sigmoid activation function
# - 5 weights connecting the hidden layer to the output neuron
# - Learning rate: 0.01
# - Single target output
# ---------------------------------------------------------------------------

# Activation functions, error, and their derivatives
def sigmoid(sop):
    return 1 / (1 + np.exp(-sop))

def error(predicted, target):
    return np.power(predicted - target, 2)

def deriv_error_predicted(predicted, target):
    return 2 * (predicted - target)

def deriv_activacion_sop(sop):
    sig = sigmoid(sop)
    return sig * (1 - sig)

def deriv_sop_w(val):
    return val  

def update_w(w, learning_rate, grad):
    return w - learning_rate * grad

# Initialize variables and weights
errors = []
iteration = []
predictions = []

num_inputs = 3
x = np.array([0.1, 0.3, 0.2])

# Weights for each of the 5 hidden neurons (each with 3 values)
w1_3 = np.random.rand(num_inputs)
w2_3 = np.random.rand(num_inputs)
w3_3 = np.random.rand(num_inputs)
w4_3 = np.random.rand(num_inputs)
w5_3 = np.random.rand(num_inputs)

# Weights for the output layer (5 weights, one for each hidden neuron)
w_out = np.random.rand(5)
w_out_old = w_out.copy()

target = np.array([0.2])
learning_rate = 0.01

print("Initial weights:")
print("w1_3:", w1_3)
print("w2_3:", w2_3)
print("w3_3:", w3_3)
print("w4_3:", w4_3)
print("w5_3:", w5_3)
print("w_out:", w_out)

for k in range(10000):
    iteration.append(k)
    # --- Forward Pass ---
    # Hidden layer (5 neurons)
    sop1 = np.sum(w1_3 * x)
    sop2 = np.sum(w2_3 * x)
    sop3 = np.sum(w3_3 * x)
    sop4 = np.sum(w4_3 * x)
    sop5 = np.sum(w5_3 * x)
    
    sig1 = sigmoid(sop1)
    sig2 = sigmoid(sop2)
    sig3 = sigmoid(sop3)
    sig4 = sigmoid(sop4)
    sig5 = sigmoid(sop5)
    
    # Array with the outputs of the hidden layer
    hidden_outputs = np.array([sig1, sig2, sig3, sig4, sig5])
    
    # Output neuron
    sop_out = np.sum(w_out * hidden_outputs)
    predicted = sigmoid(sop_out)
    err = error(predicted, target)
    
    predictions.append(predicted)
    errors.append(err)
    
    # --- Backward Pass ---
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop_out)
    
    # Update weights from hidden layer to output layer
    grad_out = np.empty(5)
    grad_out[0] = deriv_sop_w(sig1) * g2 * g1
    grad_out[1] = deriv_sop_w(sig2) * g2 * g1
    grad_out[2] = deriv_sop_w(sig3) * g2 * g1
    grad_out[3] = deriv_sop_w(sig4) * g2 * g1
    grad_out[4] = deriv_sop_w(sig5) * g2 * g1
    
    w_out = update_w(w_out, learning_rate, grad_out)
    
    # Update weights for each hidden neuron
    # Neuron 1
    g3 = deriv_sop_w(w_out_old[0]) 
    g4 = deriv_activacion_sop(sop1)
    g5 = deriv_sop_w(x)
    grad_w1 = g5 * g4 * g3 * g2 * g1
    w1_3 = update_w(w1_3, learning_rate, grad_w1)
    
    # Neuron 2
    g3 = deriv_sop_w(w_out_old[1]) 
    g4 = deriv_activacion_sop(sop2)
    g5 = deriv_sop_w(x)
    grad_w2 = g5 * g4 * g3 * g2 * g1
    w2_3 = update_w(w2_3, learning_rate, grad_w2)
    
    # Neuron 3
    g3 = deriv_sop_w(w_out_old[2]) 
    g4 = deriv_activacion_sop(sop3)
    g5 = deriv_sop_w(x)
    grad_w3 = g5 * g4 * g3 * g2 * g1
    w3_3 = update_w(w3_3, learning_rate, grad_w3)
    
    # Neuron 4
    g3 = deriv_sop_w(w_out_old[3]) 
    g4 = deriv_activacion_sop(sop4)
    g5 = deriv_sop_w(x)
    grad_w4 = g5 * g4 * g3 * g2 * g1
    w4_3 = update_w(w4_3, learning_rate, grad_w4)
    
    # Neuron 5
    g3 = deriv_sop_w(w_out_old[4]) 
    g4 = deriv_activacion_sop(sop5)
    g5 = deriv_sop_w(x)
    grad_w5 = g5 * g4 * g3 * g2 * g1
    w5_3 = update_w(w5_3, learning_rate, grad_w5)
    
    w_out_old = w_out.copy()

print("Final weights:")
print("w1_3:", w1_3)
print("w2_3:", w2_3)
print("w3_3:", w3_3)
print("w4_3:", w4_3)
print("w5_3:", w5_3)
print("w_out:", w_out)

# Plot the evolution of the prediction and error
plt.figure()
plt.plot(iteration, predictions)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.title("Evolution of the Prediction")
plt.show()

plt.figure()
plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Evolution of the Error")
plt.show()
