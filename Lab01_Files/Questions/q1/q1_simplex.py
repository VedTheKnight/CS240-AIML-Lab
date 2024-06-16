"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(a) Firstly, implement simplex method covered in class from scratch to solve the LP

simplex reference:
https://www.youtube.com/watch?v=t0NkCDigq88
"""
import numpy
import pulp
import pandas as pd
import argparse


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments



def simplex_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> list:
    """
    Implement LP solver using simplex method.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: list of pivot values simplex method encountered in the same order
    """
    pivot_value_list = []
    ################################################################
    # %% Student Code Start
    T = {f'x_{i}': [A_matrix[j,i] for j in range(len(A_matrix))] for i in range(len(c))}
    
    for i in range(len(c)):
        T[f's_{i}'] = [0 for i in range(len(A_matrix))]
        T[f's_{i}'][i] = 1
    
    T['z'] = [0 for i in range(len(b))] 
    T['b'] = [b[i] for i in range(len(b))]

    for i in range(len(c)):
        T[f'x_{i}'].append(-c[i])

    for i in range(len(c)):
        T[f's_{i}'].append(0)

    T['z'].append(1)
    T['b'].append(0)

    df = pd.DataFrame.from_dict(T)

    
    def checkOpt(bottom,n):
        for i in range(n):
            if(bottom.iloc[i] < 0):
                return True
        
        return False

    while(checkOpt(df.iloc[-1],len(c))):
        bottom = df.iloc[-1] 
        min_index = bottom.idxmin()

        b = df['b']
        b = b/df[min_index]
        b = b.apply(lambda x: numpy.inf if x < 0 else x)
        row_index = b[:-1].idxmin()
        pivot = df[min_index][row_index]
        pivot_value_list.append(pivot)

        df.iloc[row_index] = (df.iloc[row_index] / pivot)

        for i in range(len(df)):
            if(i == row_index):
                continue
            
            df.iloc[i] = df.iloc[i] - df.iloc[row_index]*df.iloc[i][min_index]

    # Implement here
    # %% Student Code End
    ################################################################

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # Read the inputs A, b, c and run solvers
    # There are 2 test cases provided to test your code, provide appropriate command line args to test different cases.
    matrix_A = pd.read_csv("{}/A.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    vector_c = pd.read_csv("{}/c.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()
    vector_b = pd.read_csv("{}/b.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()

    simplex_pivot_values = simplex_solver(matrix_A, vector_c, vector_b)
    for val in simplex_pivot_values:
        print(val)
