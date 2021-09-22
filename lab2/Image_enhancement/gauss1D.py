import numpy as np

def gauss1D(sigma , kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    
    x = np.arange(-np.floor(kernel_size/2), np.floor(kernel_size/2)+1, 1)
    G = (1/(sigma * np.sqrt(2*np.pi))) * np.exp((-x**2)/(2*sigma**2))
    #normalize
    G = G / np.sum(G)
    return G


if __name__ == '__main__':
    G = gauss1D(2, 5)
    print(G)
