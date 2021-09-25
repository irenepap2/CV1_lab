# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from createGabor import createGabor

def Visualize_parameters():
    #We set a base value for each parameter to make the comparisons
    base_theta = np.pi/4
    base_sigma = 4
    base_gamma = 0.5
    base_lamda = 3
    base_psi = 0

    base_gabor = createGabor(base_sigma, base_theta, base_lamda, base_psi, base_gamma )

    theta_values = [np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/6, 0]
    theta_names = ["180", "90", "60", "45", "30", "0"]
    sigma_values = [0, 0.1, 1, 2, 4, 6]
    gamma_values = [0.5, 0.7, 1, 1.5, 1.7, 4]

    
    count = 1
    for theta in theta_values:
        theta_gabor = createGabor(base_sigma, theta, base_lamda, base_psi, base_gamma)
        theta_name = theta_names[count-1]
        plt.subplot(1, 6, count)
        count = count +1 
        plt.imshow(theta_gabor[:, :, 1], cmap="gray")
        plt.title("θ = "+theta_name+"°")
        plt.savefig("visualization_results/theta_comparison.png")
   
    count = 1
    for sigma in sigma_values:
        sigma_gabor = createGabor(sigma, base_theta, base_lamda, base_psi, base_gamma)
        plt.subplot(1, 6, count)
        count = count +1 
        plt.imshow(sigma_gabor[:, :, 1], cmap="gray")
        plt.title("σ = " + str(sigma))
        plt.savefig("visualization_results/sigma_comparison.png")
       
    count = 1
    for gamma in gamma_values:
        gamma_gabor = createGabor(base_sigma, base_theta, base_lamda, base_psi, gamma)
        plt.subplot(1, 6, count)
        count = count +1 
        plt.imshow(gamma_gabor[:, :, 1], cmap="gray")
        plt.title("γ = " + str(gamma))
        plt.savefig("visualization_results/gamma_comparison.png")

if __name__ == '__main__':
    Visualize_parameters()