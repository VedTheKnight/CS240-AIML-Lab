import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, a * b)
    centered_X = X_flat - np.mean(X_flat, axis=0)
    cov_matrix = np.matmul(centered_X.T,centered_X)/centered_X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_components = eigenvectors[:,-k:]
    return np.fliplr(principal_components)
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    N, a, b = X.shape
    # Flatten each image into a vector
    X_flat = X.reshape(N, a * b)
    # Calculate the projection of each image onto the basis vectors
    projections = np.matmul(X_flat, basis)
    return projections
    # END TODO
    
# if __name__ == '__main__':
#     mnist_data = 'mnist.pkl'
#     with open(mnist_data, 'rb') as f:
#         data = pkl.load(f)
#     # Now you can work with the loaded object
#     X=data[0]
#     y=data[1]
#     k=10
#     basis = pca(X,k)
#     print(projection(X,basis))