import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def make_directory_structure(): 
    os.makedirs('./images/vanilla', exist_ok=True)

def plot_decision_boundary(x,y,w,name="boundary"):
    plt.figure()
    plt.scatter(x[y==-1][:,0],x[y==-1][:,1], c=['blue'])
    plt.scatter(x[y==1][:,0],x[y==1][:,1], c=['red'])
    plt.axline((0,-w[-1]/w[1]),(1,-(w[0]+w[-1])/w[1]), c='black', marker='o')
    plt.savefig(f"{name}.png")

def test_train_split(x, y, frac=0.8):
    '''
    Input: x: np.ndarray: features of shape (data_size, input_dim)
           y: np.ndarray: labels of shape (data_size,)
           frac: float: fraction of dataset to be used for test set
    Output: x_train, y_train, x_test, y_test
    '''
    cut = math.trunc(frac*x.shape[0])
    return x[:cut], y[:cut], x[cut:], y[cut:]

def hinge_loss_gradient(x_appended, y, w): 
    '''
        Input: x_appended: datapoint at which gradient is to be calculated of shape (input_dim+1,)
               y: label - scalar
               w: np.ndarray: shape (input_dim+1,)
        Output: np.ndarray: gradient of hinge loss with respect to w of shape (input_dim+1,)
    '''
    #TODO implement the gradient of hinge loss for given x, y and w

    gradient = np.zeros_like(x_appended)

    for i in range(x_appended.shape[0]):
        if y*np.dot(x_appended,w) < 1:
            gradient[i] = -1 * y * x_appended[i]
        else:
            gradient[i] = 0
    #print(gradient)
    return gradient

class Perceptron():
    def __init__(self, input_dim, lam=0.8):
        '''
            Input: input_dim: int: size of input
                   lam: float: parameter of geometric moving average. Moving average is calculated as
                            a_{t+1} = lam*a_t + (1-lam)*w_{t+1}
        '''
        self.weights = np.zeros(input_dim + 1)
        self.running_avg_weights = self.weights # redundant for this part
        self.lam = lam
    
    def fit(self, x, y, plot_flag, lr = 0.001, epochs = 50):
        '''
            Input: x: np.ndarray: training features of shape (data_size, input_dim)
                   y: np.ndarray: training labels of shape (data_size,)
                   lr: float: learning rate
                   epochs: int: number of epochs
            Output weights_history: list of np.ndarray: list of weights at each epoch
                    avg_weights_history: list of np.ndarray: list of running average weights at each epoch
        ''' 

        n,d = x.shape
        weights_history = []
        seed_everything()
        # TODO concatenate 1's at the end of x to make it of the shape (data_size, input_dim+1) so that w[-1] can be the bias term
        x = np.concatenate((x,np.ones((x.shape[0],1))),axis = 1)

        for epoch in range(epochs):
            #print(self.weights)
            for i in range(n):
                #TODO implement the weight update for each data point at a randomly chosen index (use np.random.randint()) 
                index = np.random.randint(0,x.shape[0])
                
                dw = hinge_loss_gradient(np.transpose(x[index,:]),y[index],self.weights)
                self.weights -= lr*dw

            if plot_flag:
                plot_decision_boundary(x,y,self.get_decision_boundary(False),f"images/vanilla/{epoch:05d}")  
            
            if epoch%5==0:
                print(f"Epoch: {epoch}, Decision Boundary: {self.get_decision_boundary(False)}")
                weights_history.append(self.weights)

        return weights_history

    def predict(self, x, running_avg = False):
        '''
            Input: x: np.ndarray: test features of shape (data_size, input_dim)
                   running_avg: bool: choose whether to use the running average weights for prediction
            Output: y_pred: np.ndarray: predicted labels of shape (data_size,)
        '''
        # TODO concatenate 1's at the end of x to make it of the shape (data_size, input_dim+1) so that w[-1] can be the bias term
        x = np.concatenate((x,np.ones((x.shape[0],1))),axis = 1)
        # TODO make y_pred using either the final weight vector or the moving average of the weights
        
        def signum(x):
            return 1 if x > 0 else -1
        
        sign = np.vectorize(signum)
        if running_avg:
            y_pred = sign(np.matmul(x,self.running_avg_weights))
        else:
            y_pred = sign(np.matmul(x,self.weights))
        return y_pred
    
    def get_decision_boundary(self, running_avg = False):
        '''
            Input: running_avg: bool: choose whether to use the running average weights for prediction
            Output: np.ndarray of shape (input_dim+1,) representing the decision boundary
        '''
        if running_avg:
            return self.running_avg_weights
        else:
            return self.weights


def accuracy(y_test, y_pred):
    '''
        Input: y: np.ndarray: true labels of shape (data_size,)
                y_pred: np.ndarray: predicted labels of shape (data_size,)
        Output: float: accuracy
    '''
   
   #TODO 
    accuracy = (y_test == y_pred).sum() / (y_test == y_pred).shape[0]
    return accuracy

if __name__ == "__main__":
    seed_everything()
    make_directory_structure()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="The name of the dataset to be used", required=True)
    args = parser.parse_args()

    input_dim = 2

    df = pd.read_csv(args.dataset)
    x = df[['x1', 'x2']].values
    y = df['y'].values
    x_train, y_train, x_test, y_test = test_train_split(x, y)

    p = Perceptron(input_dim)
    # TODO fit the perceptron on the train set
    weights_history = p.fit(x_train,y_train,True)
    # TODO predict on the test set using the last weight vector and print accuracy
    x_test = np.concatenate((x_test,np.ones((x_test.shape[0],1))),axis = 1)
    y_pred = np.matmul(x_test,weights_history[-1])
    
    def signum(x):
        return 1 if x > 0 else -1
        
    sign = np.vectorize(signum)
    
    y_pred = sign(y_pred)
    acc = accuracy(y_test,y_pred)
    print(f"Prediction test accuracy: {acc:.4f}")
    