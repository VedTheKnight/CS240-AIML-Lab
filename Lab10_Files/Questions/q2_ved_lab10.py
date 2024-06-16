import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        n,d,_ = X.shape
        X = X.reshape(n,-1)
        mean = np.mean(X, axis=0)

        Sb = np.zeros((d**2, d**2))
        Sw = np.zeros((d**2, d**2))

        for i in np.unique(y):
            mean_i = np.mean(X[y == i], axis=0)
            Sw += (X[y == i] - mean_i).T.dot(X[y == i] - mean_i)
            mean_difference = (mean_i - mean).reshape(-1, 1)
            Sb += X[y == i].shape[0] * mean_difference.dot(mean_difference.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

        eig_pairs = []
        for i in range(len(eigenvalues)):
            eig_pairs.append((np.abs(eigenvalues[i]), eigenvectors[:,i]))
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        linear_discriminants = []
        for i in range(0, self.n_components):
            linear_discriminants.append(eig_pairs[i][1].reshape(d**2, 1))
        self.linear_discriminants = np.hstack(linear_discriminants)
        return self.linear_discriminants                
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        projected = np.dot(X.reshape(X.shape[0], -1), w)
        return projected                   
        # END TODO
    
if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open(mnist_data, 'rb') as f:
        data = pkl.load(f)
    X=data[0]
    y=data[1]
    k=10
    lda = LDA(k)
    w=lda.fit(X, y)
    X_lda = lda.transform(X,w)