import sys
import os
import skimage.io
import skimage.filters
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
import pandas as pd
from numba import jit
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import cv2
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def make_and_visualise_in_silico_data():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    movie = np.stack((first_frame, second_frame))
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','small_in_silico_data.mp4'),dpi=300) 
    
def first_try_variational_flow_method():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    movie = np.stack((first_frame, second_frame))
    
    iterations = 1000
    iteration_stepsize = 100
    stepsizes = np.arange(iteration_stepsize,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1,
                                                           v_x_guess=1.1,
                                                           v_y_guess=1.1,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = False)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'variational_test_iterations' + str(iterations) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.05,
                                             arrow_boxsize = 4)
    
    fig = plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.title('Iteration ' + str(stepsizes[i]))
        this_speed_frame = np.zeros((movie.shape[1], movie.shape[2]))
        this_speed_frame[:,:] = result['speed_steps'][0,i,:,:]
        optical_flow.costum_imshow(this_speed_frame,delta_x = delta_x, autoscale = True, cmap = 'viridis')
        colorbar = plt.colorbar()
        plt.clim(np.min(result['speed_steps']),np.max(result['speed_steps']))
        colorbar.ax.set_ylabel('Motion speed [$\mathrm{\mu m}$/s]')
    ani = FuncAnimation(fig, animate, frames=result['speed_steps'].shape[1])
    ani.save(os.path.join(os.path.dirname(__file__),'output','variational_iterations_' + str(iterations) + '.mp4'),dpi=300) 
 
    error_measure = np.zeros(iterations//iteration_stepsize - 1)
    for iteration_index in range(iterations//iteration_stepsize - 1):
        this_error_measure = (np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:] - 
                                             result['speed_steps'][0,iteration_index,:,:])/
                         np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:]))
        error_measure[iteration_index] = this_error_measure
        
    plt.figure(figsize = (2.5,2.5), constrained_layout = True) 
    plt.plot(stepsizes[1:], error_measure)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('relative step size')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','variational_stepsizes.pdf')) 

if __name__ == '__main__':
    # make_and_visualise_in_silico_data()
    first_try_variational_flow_method()
