import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standardize(column):
    return (column - column.mean()) / column.std()

def covariance(col1, col2):
    adjc1 = col1 - col1.mean()
    adjc2 = col2 - col2.mean()
    return np.dot(adjc1,adjc2.transpose())[0,0]/len(col1)


def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA


    # first standardize the dataframe
    for i in range(len(init_array.columns)):
        init_array.iloc[:,[i]] = init_array.iloc[:,[i]].apply(standardize)


    # compute covariance matrix
    covariance_matrix = np.zeros((len(init_array.columns),len(init_array.columns)))
    # compute the covariance matrix for the dataset
    for i in range(len(init_array.columns)):
        for j in range(len(init_array.columns)):
            covariance_matrix[i,j] = covariance(init_array.iloc[:,[i]],init_array.iloc[:,[j]])

            
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    group = [(eigenvalues[i], eigenvectors[:,i]) for i in range(len(init_array.columns))]
    sorted_group = sorted(group, key=lambda x: x[0], reverse = True)
    

    sorted_eigenvalues = [round(eig,4) for eig,_ in sorted_group]
    chosen_eigenvectors = [second for _,second in sorted_group[:dimensions]]
    
    chosen_eigenvectors = np.stack(chosen_eigenvectors,axis = 0).transpose()
    final_data = np.matmul(init_array.to_numpy(),chosen_eigenvectors)
    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[:,0],final_data[:,1])
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.savefig('out.png')
    # END TODO
