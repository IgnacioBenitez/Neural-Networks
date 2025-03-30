import numpy
import matplotlib.pyplot as plt

# Simple Perceptron Training with Two Inputs
# Architecture: 
# - 2 input neurons (x1, x2)
# - 1 neuron with a sigmoid activation function
# - 2 weights (w1, w2)
# - Learning rate: 0.01
# ---------------------------------------------------------------------------

# STEP 1 --> Define the training functions for the network
# Forward functions
# Backward functions

def sigmoid(sop):
    return 1/(1 + numpy.exp(-1*sop))

def error(predicted, target):
    return numpy.power(predicted - target, 2)

# Backward functions
def deriv_error_predicted(predicted, target):
    return 2 * (predicted - target)

def deriv_activacion_sop(sop):
    return sigmoid(sop) * (1 - sigmoid(sop))

def deriv_sop_w(x):
    return x

def update_w(w, learning_rate, grad):
    return w - learning_rate * grad

# STEP 2 --> Initialize values

errors = []
iteration = []
prediction = []
x1 = 0.1
x2 = 0.4
w1 = numpy.random.rand()
w2 = numpy.random.rand()

print("Initial W1: ", w1)
print("Initial W2: ", w2)

target = 0.3
learning_rate = 0.01 # Hyperparameter

for k in range(60000):
    iteration.append(k)
    # STEP 3: Forward Pass

    sop = x1 * w1 + x2 * w2
    predicted = sigmoid(sop)
    err = error(predicted, target)

    errors.append(err)
    prediction.append(predicted)

    # STEP 4: Backward Pass
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop)
    g3 = deriv_sop_w(x1)
    g4 = deriv_sop_w(x2)

    grad1 = g1 * g2 * g3
    grad2 = g1 * g2 * g4

    w1 = update_w(w1, learning_rate, grad1)
    w2 = update_w(w2, learning_rate, grad2)

print("Final weight w1: ", w1)
print("Final weight w2: ", w2)

plt.plot(iteration, prediction)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.show()

plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.show()
