import numpy as np

def linear_kernel(X,Y,pars=[1]):
    return (np.dot(X,Y.T)+pars[0])

def polynomial_kernel(X,Y,pars=[1,3,1]):
    return ( (pars[0] * np.dot(X,Y.T) + pars[2])**pars[1] )

def gaussian_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.exp( -np.sum( (x - y)**2 )  / ( 2*(pars[0]**2) ) ) 

    return gram_matrix

def exponential_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.exp( -np.sqrt(np.sum( (x - y)**2 ))  / ( 2*(pars[0]**2) ) ) 

    return gram_matrix

def hyperbolic_tangent_kernel(X,Y,pars=[1,1]):
    return ( np.tanh(pars[0]* np.dot(X,Y.T)+pars[1]) )

def rational_quadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = 1 - (np.sum( (x - y)**2 )) / ( np.sum( (x - y)**2 ) + pars[0] )

    return gram_matrix

def multiquadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sqrt(np.sum( (x - y)**2 ) + pars[0]**2)

    return gram_matrix

def inverse_multiquadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = 1/(np.sqrt(np.sum( (x - y)**2 ) + pars[0]**2))

    return gram_matrix

def log_kernel(X,Y,pars=[3]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = - np.log( np.sum( (x - y)**2 ) ** (pars[0]/2) + 1)

    return gram_matrix

def power_kernel(X,Y,pars=[3]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = - np.sum( (x - y)**2 ) ** (pars[0]/2)

    return gram_matrix
