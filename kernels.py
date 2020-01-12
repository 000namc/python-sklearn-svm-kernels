import numpy as np


##############################################################################
######################   dot product kernels     #############################

def linear_kernel(X,Y,pars=[0]):
    return (np.dot(X,Y.T))

def polynomial_kernel(X,Y,pars=[1]):
    return ( np.dot(X,Y.T)**pars[0] )

def hyperbolic_tangent_kernel(X,Y,pars=[0]):
    return ( np.tanh(np.dot(X,Y.T)) )

def vovks_real_polynomial_kernel(X,Y,pars=[5]):
    return ( (1-np.dot(X,Y.T) ** (pars[0])) / (1-np.dot(X,Y.T)) )
    
def vovks_infinite_polynomial_kernel(X,Y,pars=[0]):
    return ( 1 / (1-np.dot(X,Y.T)) )
    
##############################################################################
######################   stationary kernels     ##############################

def gaussian_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )
            gram_matrix[i, j] = np.exp( - norm_square / ( 2*(pars[0]**2) ) ) 

    return gram_matrix

def laplacian_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )
            gram_matrix[i, j] = np.exp( - np.sqrt(norm_square) / ( 2*(pars[0]**2) ) ) 

    return gram_matrix

def rational_quadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )            
            gram_matrix[i, j] = 1 - (norm_square) / ( norm_square + pars[0] )

    return gram_matrix

def multiquadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )
            gram_matrix[i, j] = np.sqrt(norm_square + pars[0]**2)

    return gram_matrix

def inverse_multiquadratic_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )            
            gram_matrix[i, j] = 1 / (np.sqrt( norm_square + pars[0]**2))

    return gram_matrix

def circular_kernel(X,Y,pars = [1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            
            norm_square = np.sum( (x - y)**2 )            
            if np.sqrt(norm_square) < pars[0]:
                gram_matrix[i,j] = (2/np.pi) * np.arccos(- np.sqrt(norm_square)/pars[0]) - (2/np.pi) * (np.sqrt(norm_square)/pars[0]) * np.sqrt(1-(np.sqrt(norm_square)/pars[0])**2)
                
    return gram_matrix

def spherical_kernel(X,Y,pars = [1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            
            norm_square = np.sum( (x - y)**2 )            
            if np.sqrt(norm_square) < pars[0]:
                gram_matrix[i,j] = 1 - (3/2) * (np.sqrt(norm_square)/pars[0]) + (1/2) * (np.sqrt(norm_square)/pars[0])**3 
                
    return gram_matrix
    
    

def wave_kernel(X,Y,pars=[1]):
    gram_matrix = np.ones((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            
            norm_square = np.sum( (x - y)**2 )         
            if norm_square != 0:
                gram_matrix[i,j] = (pars[0] / (np.sqrt(norm_square))) * np.sin(np.sqrt(norm_square)/pars[0])
            
    return gram_matrix
    


def power_kernel(X,Y,pars=[3]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )
            gram_matrix[i, j] = - norm_square ** (pars[0]/2)

    return gram_matrix

def log_kernel(X,Y,pars=[3]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )            
            gram_matrix[i, j] = - np.log( norm_square ** (pars[0]/2) + 1)

    return gram_matrix

def generalized_tstudent_kernel(X,Y,pars =[1]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            norm_square = np.sum( (x - y)**2 )            
            gram_matrix[i, j] = 1 / (1 + norm_square ** (pars[0]/2))

    return gram_matrix
    


##############################################################################
######################   other kernels     ###################################

def anova_kernel(X,Y,pars=[2]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i,j] = (sum(np.exp(-pars[0] * (x-y)**2)))**2

    return gram_matrix

def spline_kernel(X,Y,pars=[0]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i,j] = np.prod( 1 + x*y + x*y * np.minimum(x,y) - ((x+y)/2) * (np.minimum(x,y)**2) + (np.minimum(x,y)**3)/3 )
            
    return gram_matrix
def chi_square_kernel(X,Y,pars=[0]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i,j] = 1 - sum( 2*x*y / (x+y))

    return gram_matrix
def histogram_intersection_kernel(X,Y,pars=[0]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i,j] = sum(np.minimum(x,y))

    return gram_matrix
def hellingers_kernel(X,Y,pars=[0]):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i,j] = sum(np.sqrt(x*y))

    return gram_matrix



