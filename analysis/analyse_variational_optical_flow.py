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
import tifffile

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def make_and_visualise_in_silico_data():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    tifffile.imwrite(os.path.join(os.path.dirname(__file__),'output','in_silico_frame_0.tif'),first_frame ) 
    tifffile.imwrite(os.path.join(os.path.dirname(__file__),'output','in_silico_frame_1.tif'),second_frame ) 
    print(delta_x)
    print(first_frame -second_frame)
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
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.51, y_position = 2.51, sigma = 3, width = 5, dimension = 50)
    movie = np.stack((first_frame, second_frame))
    
    # iterations = 10000
    # iteration_stepsize = 500
    iterations = 10000
    iteration_stepsize = 500
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=0.1,
                                                           v_x_guess=0.07,
                                                           v_y_guess=0.07,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = True,
                                                           include_remodelling = False)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'variational_test_iterations_0.1_' + str(iterations) + '.mp4'), 
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
    ani.save(os.path.join(os.path.dirname(__file__),'output','variational_iterations_0.1_' + str(iterations) + '.mp4'),dpi=300) 
 
    error_measure = np.zeros(iterations//iteration_stepsize)
    for iteration_index in range(iterations//iteration_stepsize):
        this_error_measure = (np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:] - 
                                             result['speed_steps'][0,iteration_index,:,:])/
                         np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:]))
        error_measure[iteration_index] = this_error_measure
        
    plt.figure(figsize = (2.5,2.5), constrained_layout = True) 
    plt.plot(stepsizes[1:], error_measure)
    plt.title('Stepsize per ' + str(iteration_stepsize) + '\niterations')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('relative step size')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','variational_stepsizes_0.1_' + str(iterations) + '.pdf')) 

def reproduce_matlab_variational_flow_method():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.51, y_position = 2.51, sigma = 3, width = 5, dimension = 50)
    first_frame = first_frame/np.max(first_frame)*255
    second_frame = second_frame/np.max(second_frame)*255
    movie = np.stack((first_frame, second_frame))
    
    delta_x = 1
    # iterations = 10000
    # iteration_stepsize = 500
    iterations = 1000
    iteration_stepsize = 100
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.003,
                                                           v_y_guess=0.003,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = True,
                                                           include_remodelling = False)
    
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
 
    error_measure = np.zeros(iterations//iteration_stepsize)
    for iteration_index in range(iterations//iteration_stepsize):
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

    # This tests that the mean speed comes out after 1000 iterations like this:
    assert(np.mean(result['speed']-0.08600834591294404 <1e-15))

def reproduce_matlab_example_vortex_pair():
    first_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_1.tif')) 
    second_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_2.tif')) 

    movie = np.stack((first_frame, second_frame))
    
    delta_x = 1.0

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','liu_shen_vortex__in_silico_data.mp4'),dpi=300) 
 
    # iterations = 10000
    # iteration_stepsize = 500
    iterations = 100
    iteration_stepsize = 10
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = 1.0,
                                                           delta_t = 1.0,
                                                           alpha=2000,
                                                           v_x_guess=0.015,
                                                           v_y_guess=0.015,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = True,
                                                           include_remodelling = False)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'vortex_test_iterations' + str(iterations) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.05,
                                             arrow_boxsize = 20)
    
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
    ani.save(os.path.join(os.path.dirname(__file__),'output','vortex_iterations_' + str(iterations) + '.mp4'),dpi=300) 
 
    error_measure = np.zeros(iterations//iteration_stepsize)
    for iteration_index in range(iterations//iteration_stepsize):
        this_error_measure = (np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:] - 
                                             result['speed_steps'][0,iteration_index,:,:])/
                         np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:]))
        error_measure[iteration_index] = this_error_measure
        
    plt.figure(figsize = (2.5,2.5), constrained_layout = True) 
    plt.plot(stepsizes[1:], error_measure)
    plt.title('Stepsize per ' + str(iteration_stepsize) + '\niterations')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('relative step size')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','vortex_stepsizes.pdf')) 


    
if __name__ == '__main__':
    # make_and_visualise_in_silico_data()
    # first_try_variational_flow_method()
    # reproduce_matlab_variational_flow_method()
    reproduce_matlab_example_vortex_pair()
