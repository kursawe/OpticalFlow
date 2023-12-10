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
import time

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def make_and_visualise_in_silico_data():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.51, y_position = 2.51, sigma = 3, width = 5, dimension = 50)
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
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    movie = np.stack((first_frame, second_frame))
    
    # iterations = 10000
    # iteration_stepsize = 500
    iterations = 10000
    iteration_stepsize = 500
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.15,
                                                           v_y_guess=0.15,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = False,
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
                                                           include_remodelling = False,
                                                           use_liu_shen = True)
    
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
                                                           include_remodelling = False,
                                                           use_liu_shen = True)
    
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

def make_new_test_data_with_non_moving_boundary():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    first_frame-=0.8
    second_frame-=0.8
    first_frame[first_frame<0]=0.0
    second_frame[second_frame<0]=0.0
    movie = np.stack((first_frame, second_frame))
    
    new_movie = optical_flow.blur_movie(movie, smoothing_sigma=4)
    print(new_movie[1,:,:] - new_movie[0,:,:])
    print(new_movie[1,22:27,22:27] - movie[0,22:27, 22:27])
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(new_movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','new_in_silico_data.mp4'),dpi=300) 
 
def apply_method_to_data_with_non_moving_boundary():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    first_frame-=0.8
    second_frame-=0.8
    first_frame[first_frame<0]=0.0
    second_frame[second_frame<0]=0.0
    noise_1 = np.random.random((50,50))
    noise_2 = np.random.random((50,50))
    # first_frame = np.abs(first_frame+noise_1*1e-3)
    # second_frame = np.abs(second_frame+noise_2*1e-3)
    movie = np.stack((first_frame, second_frame))
    movie = optical_flow.blur_movie(movie, smoothing_sigma=2)
    
    iterations = 10000000
    iteration_stepsize = 500000
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.09,
                                                           v_y_guess=0.09,
                                                           remodelling_guess=0.07,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = True,
                                                           include_remodelling = True)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'variational_test_iterations_w_noise_' + str(iterations) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.5,
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
    ani.save(os.path.join(os.path.dirname(__file__),'output','variational_iterations_w_noise_' + str(iterations) + '.mp4'),dpi=300) 
 
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
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','variational_stepsizes_w_noise_' + str(iterations) + '.pdf')) 
    
def test_and_time_new_numpy_method():
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.6, y_position = 2.6, sigma = 3, width = 5, dimension = 50)
    first_frame-=0.8
    second_frame-=0.8
    first_frame[first_frame<0]=0.0
    second_frame[second_frame<0]=0.0
    noise_1 = np.random.random((50,50))
    noise_2 = np.random.random((50,50))
    first_frame = np.abs(first_frame+noise_1*1e-3)
    second_frame = np.abs(second_frame+noise_2*1e-3)
    movie = np.stack((first_frame, second_frame))
    movie = optical_flow.blur_movie(movie, smoothing_sigma=4)
    
    iterations = 1000
    start_time = time.time()
    result_old = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.095,
                                                           v_y_guess=0.095,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
                                                           use_jacobi = True,
                                                           include_remodelling = False,
                                                           use_legacy = True)
    end_time = time.time()
    time_1 = end_time - start_time

    result_new = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.095,
                                                           v_y_guess=0.095,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
                                                           use_jacobi = True,
                                                           include_remodelling = False)
 
    start_time = time.time()
    result_new = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.095,
                                                           v_y_guess=0.095,
                                                           remodelling_guess=0.05,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
                                                           use_jacobi = True,
                                                           include_remodelling = False)
    
    end_time = time.time()
    time_2 = end_time - start_time
    print('the old method took ')
    print(time_1)
    print('the new method took')
    print(time_2)
    
    print('the difference between the results is')
    print(np.linalg.norm(result_new['speed'] -result_old['speed']))
    print('the maximum element_wise difference is')
    print(np.max(result_new['speed'] -result_old['speed']))
    print('the stepsize was')
    initial_guess_x = np.zeros_like(result_new['v_x'])
    initial_guess_y = np.zeros_like(result_new['v_x'])
    initial_guess_x[:] = 0.095
    initial_guess_y[:] = 0.095
    initial_speed = np.sqrt(initial_guess_x**2 + initial_guess_y**2)
    print(np.linalg.norm(result_new['speed'] - initial_speed))
    print('the maximal element wise step was')
    print(np.max(result_new['speed'] - initial_speed))

 
def make_convergence_analysis_simple_example(v_x = 0.1, v_y = 0.2, remodelling = 0.05):
    # make data
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.5 + v_x, y_position = 2.5 + v_y, sigma = 3, width = 5, dimension = 50)
    first_frame-=0.8
    second_frame-=0.8
    first_frame[first_frame<0]=0.0
    second_frame[second_frame<0]=0.0
    noise_1 = np.random.random((50,50))
    noise_2 = np.random.random((50,50))
    # first_frame = np.abs(first_frame+noise_1*1e-3)
    # second_frame = np.abs(second_frame+noise_2*1e-3)
    movie = np.stack((first_frame, second_frame))
    movie = optical_flow.blur_movie(movie, smoothing_sigma=3)
    movie[1,:,:] +=remodelling

    iterations = 1000000
    iteration_stepsize = 50000
    filename_start = os.path.join(os.path.dirname(__file__),'output','convergence_analysis_vx_'+ "{:.2f}".format(v_x)
                                  + '_vy_' + "{:.2f}".format(v_x) + '_rmdlng_' +  "{:.2f}".format(remodelling) + '_iterations_' +str(iterations))
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(filename_start + '_data_.mp4'),dpi=300) 
 
    # call method
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           speed_alpha=1.0,
                                                           remodelling_alpha = 10000.0,
                                                           v_x_guess=0.09,
                                                           v_y_guess=0.09,
                                                           remodelling_guess=0.1,
                                                           iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           use_jacobi = True,
                                                           include_remodelling = True)
 
    # plot convergence analysis
    optical_flow.make_convergence_plots(result, filename_start = filename_start)
    
if __name__ == '__main__':
    # make_and_visualise_in_silico_data()
    # first_try_variational_flow_method()
    # reproduce_matlab_variational_flow_method()
    # reproduce_matlab_example_vortex_pair()
    # make_new_test_data_with_non_moving_boundary()
    # apply_method_to_data_with_non_moving_boundary()
    # test_and_time_new_numpy_method()
    make_convergence_analysis_simple_example(v_x = 0.1, v_y = 0.2, remodelling = 0.05)
