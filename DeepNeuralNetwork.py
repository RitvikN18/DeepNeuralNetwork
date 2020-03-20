import numpy as np 
e=10**-10
# activation function rectified linear unit on any numpy array
def relu(Z):
    return np.maximum(Z,0)

def sigmoid(Z):
    Z=-1*Z
    return 1/(1+np.exp(Z))

# this function initializes the weigth and bias parameters of each layer of the deep neural network and returns them as a dictionary
# the arguments should be a list of dimension in each layer
# the first element in the list from the argument must contain the number of features given as input
def initialize_parameters(layer_dims):
    parameters={}      # a dictionary to store the parameters of each layer
    L=len(layer_dims)  # number of layers in the deep neural network
    for l in range(1,L):
        parameters['W' + str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)]=np.zeros((layer_dims[l],1),dtype=np.float)
    return parameters

# this function executes forward propagation and returns the output for of all layers for current parameters
# this function takes the arguments as input features, parameters and a list of activation functions for each layer
'''
def forward_propagation(X, parameters, activation):
    outputs=[]   # format=[[A[0],W[0],b[0]],[A[1],W[1],b[1]],...]
    A=X
    L=len(activation)  # number of layers in the network
    for l in range(1,L):
        A_prev=A
        if(activation[l-1]=='relu'):
            Z=np.dot(parameters['W'+str(l)],A_prev) + parameters['b'+str(l)]
            A=relu(Z)
            outputs+=[A,parameters['W'+str(l)]],parameters['b'+str(l)]
        elif(activation[l-1]=='sigmoid'):
            Z=np.dot(parameters['W'+str(l)],A_prev) + parameters['b'+str(l)]
            A=sigmoid(Z)
            outputs+=[A,parameters['W'+str(l)]],parameters['b'+str(l)]
        elif(activation[l-1]=='linear'):
            Z=np.dot(parameters['W'+str(l)],A_prev) + parameters['b'+str(l)]
            A=Z
            outputs+=[A,parameters['W'+str(l)]],parameters['b'+str(l)]
    Z=np.dot(parameters['W'+str(L)],A) + parameters['b' + str(L)]
    if activation[L-1]=='relu':
        AL=relu(Z)
    elif activation[L-1]=='sigmoid':
        AL=sigmoid(Z)
    else :
        AL=Z
    outputs+=[AL,parameters['W'+str(L)]],parameters['b'+str(L)]
    return AL,outputs
'''


def linear_forward(A, W, b):
    Z = np.dot(W, A)+b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A= relu(Z)
        activation_cache=Z
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

# takes arguments as AL- corresponding to the predicted values
#                    Y - corresponding to the actual output values in the dataset
def compute_cost(AL,Y):
    m = Y.shape[1]      #number of training examples
    cost = -(np.dot(Y,np.log(AL.T+e)) + np.dot((1-Y),np.log(1-AL.T+e)))/m
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

# function to calculate backward propagation parameters 
# return the derivative values in a dictionary
'''def backward_propagation(AL,Y,outputs,activation):
    grads={}       # a dictionary containing all the gradient values
    L=len(outputs)
    m=AL.shape[1]
    Y.reshape(AL.shape)
    dAL= -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    output=outputs[L-1]
    if activation[L-1]=='sigmoid':
        dZ=AL-Y
        A_prev,W,b=output
        dW = np.dot(dZ,A_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev = np.dot(W.T,dZ)
    elif activation[L-1]=='relu':
        if output[0]<=0:
            dZ=0
        else:
            dZ=output[0]
        A_prev,W,b=output
        dW = np.dot(dZ,A_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev = np.dot(W.T,dZ)
    else :
        dZ=(AL-Y)/(AL*(1-AL))
        A_prev, W, b = output
        dW = np.dot(dZ, A_prev.T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev,dW,db
    dA_prev_temp=dA_prev
    for l in reversed(range(L-1)):
        output=outputs[l]
        if activation[l]=='sigmoid':
            dZ=output[0]-Y
            A_prev,W,b=output
            dW_temp = np.dot(dZ,A_prev.T)/m
            db_temp = np.sum(dZ,axis=1,keepdims=True)/m
            dA_prev_temp = np.dot(W.T,dZ)
        elif activation[l]=='relu':
            if output[0]<=0:
                dZ=0
            else:
                dZ=output[0]
            A_prev,W,b=output
            dW_temp = np.dot(dZ,A_prev.T)/m
            db_temp = np.sum(dZ,axis=1,keepdims=True)/m
            dA_prev_temp = np.dot(W.T,dZ)
        else :
            dZ = (A-Y)/(A*(1-A))
            A_prev, W, b = output
            dW_temp = np.dot(dZ, A_prev.T)/m
            db_temp = np.sum(dZ, axis=1, keepdims=True)/m
            dA_prev_temp = np.dot(W.T, dZ)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
'''


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        #dZ=np.random.randn(linear_cache[0].shape)
        dz=[]
        for i in range(activation_cache.shape[0]):
            for j in range(activation_cache.shape[1]):
                if activation_cache[i][j]<=0:
                    dz+=[0]
                else :
                    dz+=[activation_cache[i][j]]
        dZ=np.array(dz)
        dZ=np.reshape(dZ,activation_cache.shape)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        A=sigmoid(activation_cache)
        dZ = dA*(A)*(1-A)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = linear_activation_backward(dAL, caches[L - 1], "sigmoid")
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" +str(L)] = current_cache[0], current_cache[1], current_cache[2]
    dA_prev_temp = current_cache[0]
    for l in reversed(range(L-1)):
        current_cache = linear_activation_backward(dA_prev_temp, caches[l], "relu")
        dA_prev_temp, dW_temp, db_temp = current_cache[0], current_cache[1], current_cache[2]
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# arguments for this function are a dictionary which contains parameters, dictionary containing gradient values and learning_rate of the model
# this function returns the updated parameters after gradient descent 
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters) //2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]
    return parameters
        
 
# the arguments are 
# X- a numpy array consistin of the features
# Y- the outputs of the given dataset X
# layer_dims- it a list of dimension in each layer
# epochs- number of iterations to be performed 
def model(X,Y,layer_dims,epochs):
    parameters = initialize_parameters(layer_dims)
    for p in range(0, epochs):
        AL, outputs = L_model_forward(X, parameters)
        print("Cost after", p+1, "th iteration:", compute_cost(AL, Y))
        grads = L_model_backward(AL, Y, outputs)
        parameters = update_parameters(parameters, grads, 0.01)
    return parameters

def predict(X,parameters):
    L = len(parameters) // 2
    AL, outputs = L_model_forward(X, parameters)
    return AL




