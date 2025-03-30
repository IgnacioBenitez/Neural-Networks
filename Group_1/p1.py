import numpy
# A simple perceptron with one input, one neuron, and sigmoid activation.

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

x = 0.1
target = 0.3
learning_rate = 0.1 # Hyperparameter 

w = numpy.random.rand()

print("Initial W: ", w)

# STEP 3: Forward Pass
sop = x * w
predicted = sigmoid(sop)
err = error(predicted, target)
print("Error ", err)

# STEP 4: Backward Pass
g1 = deriv_error_predicted(predicted, target)
g2 = deriv_activacion_sop(sop)
g3 = deriv_sop_w(x)

grad = g1 * g2 * g3

w = update_w(w, learning_rate, grad)
print("New weight ", w)

# Reduce error
sop = x * w
predicted = sigmoid(sop)
err = error(predicted, target)
print("New Error ", err)

#print(numpy.__version__)
