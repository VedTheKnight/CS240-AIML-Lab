import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def train_test_split(dataframe):
    return dataframe.iloc[0:240], dataframe.iloc[240:300]


def w_closed_form(X, Y):
    '''
    @params
        X : 2D tensor of shape(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of shape(n,1)
    calculates w_closed : 1D tensor of shape(d,1)
    writes the w_closed as a numpy array into the text file "w_closed.txt"
    returns w_closed
    '''

    # Student code start TASK 1 : Write w_closed in form of X, Y matrices
    # Round w_closed upto 4 decimal places
    
    XtX_inv = torch.inverse(torch.matmul(X.transpose(0,1),X))
    w_closed = np.matmul(XtX_inv,np.matmul(X.transpose(0,1),Y))

    
    # Student code end

    w_closed = w_closed.detach().numpy().squeeze(axis=1)
    np.savetxt('w_closed.txt', w_closed, fmt="%f")
    return w_closed

def transform_features(X, degree=1):
    '''
    For Q3
    Args:
    - X: Array containing the feature vectors.
    - degree : The degree of the polynomial to which the features are to be transformed
    
    Returns:
    - phi_X : Array containing the feature vectors with the transformed features concatenated
    '''
    #Implement the polynomial basis function transformation, and return it
    phi_X = None
    
    # Student code start TASK 1 : Write the code for polynomial basis function transformation

    def phi(x,n):
        return pow(x,n)

    phi_list = [[phi(X[i],j) for j in range(degree+1)] for i in range(len(X))]
    
    phi_X = torch.tensor(phi_list)
    #print(phi_X)

    # Student code end
    
    return phi_X


def l2_loss(X, Y, w):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
    return loss : np.float64 : scalar real value
    '''

    w = w.double()

    # Student code start TASK 2 : Write l2-loss in form of X, Y, w matrices
    # Please take care of normalization factor 1/n

    loss = (Y - torch.matmul(X,w)) ** 2
    loss = torch.mean(loss)

    # Student code end

    return (loss)


def l2_loss_derivative(X, Y, w):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
    return derivative : 1D tensor of size(d,1)
    '''

    w = w.double()

    # Student code start TASK 3 : Write l2-loss-derivative in form of X, Y, w matrices
    # Please take care of normalization factor 1/n

    error = (torch.matmul(X,w) - Y) # this is n x 1

    derivative = 2 * torch.matmul(X.transpose(0,1), error) / X.size(0)

    # Student code end

    return (derivative)


def train_model(X_train, Y_train, X_test, Y_test, w, eta):
    '''
    @params
        X_train : 2D tensor of size(n,d) over which model is trained
        n : no of samples for the X_train dataset
        d : dimension of each sample vector x(i)
        Y_train : 1D tensor of size(n,1) over which model is trained
        w : initial weights vector (that needs to be optimised using gradient descent)
        eta : learning rate
    @returns
        w : 1D tensor of size(d,1) ,  the final optimised w
        iters : Total iterations it take for algorithm to converge
        test_err : python list containing the l2-loss at each iteration

    '''

    epsilon = 1e-15  # Stopping precision
    old_loss = 0
    test_err = []  # Initially empty list
    iters = 0

    '''
    stopping condition: abs(new_loss - old_loss) <= epsilon

    Pseudo code:

    while stopping condition not met:    
        calculate old loss
        calculate gradient (dw)
        update w = w - eta*dw
        append test error to test_err (l2_loss)
    
    '''

    # Student code start TASK 4 : Write the code for gradient descent as described above

    new_loss = l2_loss(X_train,Y_train,w)
    test_err.append(new_loss)
    while(not abs(new_loss - old_loss) <= epsilon):
        old_loss = l2_loss(X_train,Y_train,w)
        print(old_loss)

        dw = l2_loss_derivative(X_train,Y_train,w)

        w = w - eta*dw

        new_loss = l2_loss(X_train,Y_train,w)

        iters+=1
        print(iters)
        print(new_loss)
        test_err.append(l2_loss(X_test,Y_test,w))

    # Student code end

    return w, test_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The name of the dataset to be used" )
    parser.add_argument('--seed', type = int, default = 335)
    parser.add_argument('--eta', type=float, default=1e-3)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    data = pd.read_csv(args.dataset , index_col=0)

    data_train, data_test = train_test_split(data)

    X_train = (data_train.iloc[:,:-1].to_numpy())
    Y_train = (data_train.iloc[:,-1].to_numpy())
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train).unsqueeze(1)

    X_test = (data_test.iloc[:,:-1].to_numpy())
    Y_test = (data_test.iloc[:,-1].to_numpy())
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test).unsqueeze(1)

    possible_degrees = range(1, 11)


    # UNCOMMENT & RUN THE CODE BELOW AFTER COMPLETING  function transform_features(X, degree=1)

    # with open('output.txt', 'w') as file:
    #     best = 0
    #     best_metric = np.inf
    #     for degree in possible_degrees:
    #         transformed_X_train = transform_features(X_train, degree)
    #         transformed_X_test = transform_features(X_test, degree)
    #         d = transformed_X_train.shape[1]
    #         # w = torch.randn(d,1)
    #         # eta = args.eta
    #         # w_trained, test_err = train_model(X_train, Y_train, X_test, Y_test, w, eta)
    #         w_closed = torch.from_numpy(w_closed_form(transformed_X_train,Y_train)).unsqueeze(1)    # closed form solution for w
    #         l2_loss_train = float(l2_loss(transformed_X_train,Y_train, w_closed))
    #         l2_test_loss = float(l2_loss(transformed_X_test,Y_test,w_closed))

    #         metric = (l2_loss_train*0.2 + l2_test_loss*0.8)*(degree**0.5) #multiply by degree to avoid overfitting
    #         if metric < best_metric:
    #             best_metric = metric
    #             optimal_degree = degree

    #         # Write some content to the file
    #         file.write(f"{degree} {l2_loss_train} {l2_test_loss}\n")

    optimal_degree = 3

    print("optimal degree",optimal_degree)






