import numpy
import matplotlib.pyplot as plt

# Training a Simple Perceptron with Iterations
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
x = 0.1
target = 0.3
w = numpy.random.rand()
print("Initial W: ", w)

learning_rate = 0.1 # Hyperparameter
for k in range(60000):
    iteration.append(k)

    # STEP 3: Forward Pass
    sop = x * w
    predicted = sigmoid(sop)
    err = error(predicted, target)

    errors.append(err)
    prediction.append(predicted)

    # STEP 4: Backward Pass
    g1 = deriv_error_predicted(predicted, target)
    g2 = deriv_activacion_sop(sop)
    g3 = deriv_sop_w(x)

    grad = g1 * g2 * g3

    w = update_w(w, learning_rate, grad)

print("Final weight ", w)

plt.plot(iteration, prediction)
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.show()

plt.plot(iteration, errors)
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.show()
