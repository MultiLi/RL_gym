import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    z = 2 * x / 3
    return 1.7159 * z / (1 + z**2/(3 + z** 2 /(5 + z**2 /(7 + z**2/ (9 + z**2 / 11)))))

def relu(x):
    return x * (x > 0)

def dsigmoid(x):
    return x* (1-x)

def dtanh(x):
    return 1.7159 * 2/3 *(1- (x/1.7159)**2)

def drelu(x):
    return (x>0)

def softmax(x):
    score = np.exp(x)
    return score / (np.sum(score,axis = 1).reshape(-1,1))

def forward(name,x):
    return


types = set(['tanh','sigmoid','relu'])
