from gauss1D import *
import numpy as np

def gauss2D(sigma_x, sigma_y , kernel_size):
    Gx = gauss1D(sigma_x, kernel_size)
    Gy = gauss1D(sigma_y, kernel_size)
    G = np.outer(Gx, Gy)
    #normalize
    G = G / np.sum(G)
    return G

if __name__ == '__main__':
    G = gauss2D(2, 2, 5)
    print(G)