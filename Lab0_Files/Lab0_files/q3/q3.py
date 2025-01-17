from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    result = v*(t - (1 - np.exp(-k*t))/k)

    return result
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    x_data = np.array(df['t'])

    y_data = np.array(df['S'])

    popt, pcov = curve_fit(func, x_data,y_data)
    
    v = popt[0]
    k = popt[1]

    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    
    x_data = np.array(df['t'])
    y_data = np.array(df['S'])
    y_fit = func(x_data,v,k)
    plt.plot(x_data,y_data,'*', label = 'data')
    plt.plot(x_data,y_data,'-', label = f"fit : v={v}, k={k}")
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('S') 
    plt.savefig('fit_curve.png')
    # END TODO
