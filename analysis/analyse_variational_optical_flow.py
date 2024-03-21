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

   
def make_fake_data(v_x = 0.1, v_y = 0.2, remodelling = 0.05, add_remodelling_slope = False):
    """ Use this method to make and return facke data in the same way between tests."""
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.5 + v_x, y_position = 2.5 + v_y, sigma = 3, width = 5, dimension = 50)
    first_frame-=0.8
    second_frame-=0.8
    first_frame[first_frame<0]=0.0
    second_frame[second_frame<0]=0.0
    first_frame/=np.max(first_frame)
    second_frame/=np.max(second_frame)
    movie = np.stack((first_frame, second_frame))
    movie = optical_flow.blur_movie(movie, smoothing_sigma=3)
    if add_remodelling_slope:
        row = np.linspace(0.0,remodelling,movie.shape[2])
        remodelling_matrix = np.tile(row,(movie.shape[1],1))
        movie[1,:,:] += remodelling_matrix
    else:
        movie[1,:,:] += remodelling
    return movie, delta_x

def make_convergence_analysis_simple_example(v_x = 0.1, v_y = 0.2, remodelling = 0.05, v_x_start = 0.09, v_y_start = 0.09,
                                             remodelling_start = 0.1):
    # make data
    movie, delta_x = make_fake_data(v_x = v_x, v_y = v_y, remodelling = remodelling)
    iterations = 400000
    iteration_stepsize = 20000
    filename_start = os.path.join(os.path.dirname(__file__),'output','convergence_analysis_vx_'+ "{:.2f}".format(v_x)
                                  + '_vy_' + "{:.2f}".format(v_x) + '_rmdlng_' +  "{:.2f}".format(remodelling) + '_iterations_' +str(iterations)
                                  + '_vx_start_' + "{:.2f}".format(v_x_start))
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
                                                           speed_alpha=10.0,
                                                           remodelling_alpha = 10000.0,
                                                           v_x_guess=v_x_start,
                                                           v_y_guess=v_y_start,
                                                           remodelling_guess=remodelling_start,
                                                           max_iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           tolerance = 1e-20,
                                                           include_remodelling = True)
 
    # plot convergence analysis
    optical_flow.make_convergence_plots(result, filename_start = filename_start)
    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
    
def try_stopping_condition():
    # make data
    movie, delta_x= make_fake_data(v_x =0.1, v_y = 0.2, remodelling = 0.05)
    max_iterations = 15000000
    tolerance = 1e-8
    # filename_start = os.path.join(os.path.dirname(__file__),'output','convergence_analysis_vx_'+ "{:.2f}".format(v_x)
                                #   + '_vy_' + "{:.2f}".format(v_x) + '_rmdlng_' +  "{:.2f}".format(remodelling) + '_iterations_' +str(iterations)
                                #   + '_vx_start_' + "{:.2f}".format(v_x)start)
    filename = os.path.join(os.path.dirname(__file__),'output','first_compound_result_with_stopping_condition.mp4')

    # call method
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           speed_alpha=1.0,
                                                           remodelling_alpha = 10000.0,
                                                           v_x_guess=0.09,
                                                           v_y_guess=0.09,
                                                           remodelling_guess=0.1,
                                                           max_iterations = max_iterations,
                                                           tolerance = tolerance,
                                                           smoothing_sigma = None,
                                                           return_iterations = False)
 
    optical_flow.make_joint_overlay_movie(result, 
                                             filename, 
                                             autoscale = True,
                                             arrow_scale = 0.5,
                                             arrow_boxsize = 4)
 
    # plot convergence analysis
    print("The total number of iterations is:")
    print(result["total_iterations"])
    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
 

def illustrate_boundary_artifacts():
    v_x = 0.1
    v_y = 0.2
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50, include_noise = False)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.5 + v_x, y_position = 2.5 + v_y, sigma = 3, width = 5, dimension = 50, include_noise = False)
    second_frame += 0.05
    movie = np.stack((first_frame, second_frame))
    
    filename_start = os.path.join(os.path.dirname(__file__),'output','boundary_example')
    
    max_iterations = 200000
    iteration_stepsize = 10000
    # max_iterations = 10
    # iteration_stepsize = 1
    # max_iterations = 90
    # iteration_stepsize = 5

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
                                                           v_x_guess=0.01,
                                                           v_y_guess=0.01,
                                                           remodelling_guess=0.00,
                                                           max_iterations = max_iterations,
                                                           iteration_stepsize = iteration_stepsize,
                                                           smoothing_sigma = None,
                                                           tolerance = 1e-17,
                                                           include_remodelling = True,
                                                           return_iterations = True)
    
    optical_flow.make_convergence_plots(result, filename_start = filename_start)

    optical_flow.make_joint_overlay_movie(result, 
                                             filename_start + '_joint_result.mp4', 
                                             autoscale = True,
                                             arrow_scale = 0.1,
                                             arrow_boxsize = 4)
 
    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
 
def reproduce_matlab_example_vortex_pair_new():
    first_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_1.tif')) 
    second_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_2.tif')) 

    movie = np.stack((first_frame, second_frame))
    movie = movie.astype(np.float64)
    
    delta_x = 1.0

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','liu_shen_vortex__in_silico_data_new.mp4'),dpi=300) 
 
    # iterations = 10000
    # iteration_stepsize = 500
    iterations = 3000
    iteration_stepsize = 150
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = 1.0,
                                                           delta_t = 1.0,
                                                           speed_alpha=2000,
                                                           remodelling_alpha=1e20,
                                                           v_x_guess=0.015,
                                                           v_y_guess=0.015,
                                                           remodelling_guess=0.00,
                                                           max_iterations = iterations,
                                                           smoothing_sigma = 0.62*6,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           include_remodelling = True)
                                                        #    use_liu_shen = True)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'vortex_test_iterations_new' + str(iterations) + '.mp4'), 
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
    ani.save(os.path.join(os.path.dirname(__file__),'output','vortex_iterations_new_' + str(iterations) + '.mp4'),dpi=300) 
 
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
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','vortex_stepsizes_new.pdf')) 


    optical_flow.make_convergence_plots(result, filename_start = os.path.join(os.path.dirname(__file__),'output','vortex_convergence_new.pdf'))

    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
    
    
    np.save(os.path.join(os.path.dirname(__file__),'output','vortex_new_convergence_result.npy'), result)
 
def analyse_emergence_of_vortex_pair_instability():

    result = np.load(os.path.join(os.path.dirname(__file__),'output','vortex_new_convergence_result.npy'),allow_pickle='TRUE').item()
    
    print("The max v_x value is at")
    print(np.unravel_index(np.argmax(result['v_x']),result['v_x'].shape))
    print("The min v_y value is at")
    print(np.unravel_index(np.argmax(-result['v_y']),result['v_y'].shape))
    
    filenamestart = os.path.join(os.path.dirname(__file__),'output','vortex_debug')

    tifffile.imsave(filenamestart + '_vx.tiff',result['v_x'])
    tifffile.imsave(filenamestart + '_vy.tiff',result['v_y'])
    tifffile.imsave(filenamestart + '_speed.tiff',result['speed'])
    
    blurred_data = optical_flow.blur_movie(result['original_data'], smoothing_sigma = 0.62*6)
    previous_frame_w_border = blurred_data[0,:,:]
    current_frame_w_border = blurred_data[1,:,:]
    previous_frame = previous_frame_w_border[1:-1,1:-1]
    current_frame = current_frame_w_border[1:-1,1:-1]

    dIdx = optical_flow.apply_numerical_derivative(previous_frame_w_border,'dx')#dI/dx_ij  #h=delta_x in equation
    dIdy = optical_flow.apply_numerical_derivative(previous_frame_w_border,'dy')#dI/dx_ij  #h=delta_x in equation
            
    dIdx_t=(current_frame_w_border[2:,1:-1] -current_frame_w_border[:-2,1:-1]
                -previous_frame_w_border[2:,1:-1] +previous_frame_w_border[:-2,1:-1])/2

    dIdy_t=(current_frame_w_border[1:-1,2:] -current_frame_w_border[1:-1,:-2]
                -previous_frame_w_border[1:-1,2:] +previous_frame_w_border[1:-1,:-2])/2

    dIdt_w_border = (current_frame_w_border
                      -previous_frame_w_border)
    dIdt = dIdt_w_border[1:-1,1:-1]

    dIdxx = optical_flow.apply_numerical_derivative(previous_frame_w_border, 'dxx')
    dIdyy = optical_flow.apply_numerical_derivative(previous_frame_w_border, 'dyy')
    dIdyx = optical_flow.apply_numerical_derivative(previous_frame_w_border, 'dyx')

    speed_alpha = 2000
    remodelling_alpha = 1e20
    ####Set up all boundary conditions
    ## LHS Matrix for the bulk
    A11 = (previous_frame*dIdxx -2*previous_frame**2 -4*speed_alpha)
    # A11 = (previous_frame*dIdxx -2*previous_frame**2)
    A12 = previous_frame*dIdyx
    A22 = previous_frame*dIdyy -2*previous_frame**2-4*speed_alpha
    # A22 = previous_frame*dIdyy -2*previous_frame**2
    A31 = -dIdx
    A32 = -dIdy
    A33 = +(1 +4*remodelling_alpha)
    
    # This is not the actual determinant, but the 2D top left sub-determinant
    det_A = A11*A22 - A12*A12

    # inverse of the matrix
    inv_A11 = A22/det_A
    inv_A12 = -A12/det_A
    inv_A22 = A11/det_A
    inv_A31 = (A32*A12 - A22*A31)/(det_A*A33)
    inv_A32 = (A12*A31 - A32*A11)/(det_A*A33)
    inv_A33 = 1/A33
    
    print('max and min det a')
    print(np.max(det_A))
    print(np.min(det_A))

    print('max and min A11')
    print(np.max(np.abs(A11)))
    print(np.min(np.abs(A11)))
 
    print('max and min A22')
    print(np.max(np.abs(A22)))
    print(np.min(np.abs(A22)))
 
####
#### These are functions I played around with while figuring things out. I may have changed the interface in optical_flow.py too much
#### for these to still work, but I figured it might be good to keep them for now.
####
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
                                                           max_iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
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
    iterations = 10000
    iteration_stepsize = 500
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    result = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = 1.0,
                                                           delta_t = 1.0,
                                                           speed_alpha=2000,
                                                           v_x_guess=0.015,
                                                           v_y_guess=0.015,
                                                           remodelling_guess=0.05,
                                                           max_iterations = iterations,
                                                           smoothing_sigma = 0.62*6,
                                                           return_iterations = True,
                                                           iteration_stepsize = iteration_stepsize,
                                                           include_remodelling = False,
                                                           tolerance = 1e-20,
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

    optical_flow.make_convergence_plots(result, filename_start = os.path.join(os.path.dirname(__file__),'output','vortex_convergence_old_new.pdf'))

    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
 
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
                                                           max_iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
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
                                                           max_iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
                                                           include_remodelling = False)
 
    start_time = time.time()
    result_new = optical_flow.conduct_variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           alpha=1.0,
                                                           v_x_guess=0.095,
                                                           v_y_guess=0.095,
                                                           remodelling_guess=0.05,
                                                           max_iterations = iterations,
                                                           smoothing_sigma = None,
                                                           return_iterations = False,
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
 
def identify_non_uniform_remodelling_rate():
    # make data
    movie, delta_x = make_fake_data(v_x = 0.05, v_y = 0.1, remodelling = 0.05, add_remodelling_slope= True)
    # perturbation, _ = optical_flow.make_fake_data_frame(x_position = 4.0, y_position = 1.0, sigma = 1.0, width = 5, dimension = 50)
    # perturbation -= 0.8
    # perturbation[perturbation<0] = 0.0
    # perturbation = skimage.filters.gaussian(perturbation, sigma =2, preserve_range = True)
    # perturbation *= 0.01/np.max(perturbation)
    # movie[1,:,:] += perturbation
    max_iterations = 1000000
    filename_start = os.path.join(os.path.dirname(__file__),'output','non_uniform_remodelling_rate')

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
                                                           speed_alpha=100.0,
                                                           remodelling_alpha = 500.0,
                                                           v_x_guess=0.05,
                                                           v_y_guess=0.1,
                                                           remodelling_guess=0.05,
                                                           max_iterations = max_iterations,
                                                           smoothing_sigma = None,
                                                           tolerance = 1e-9,
                                                           include_remodelling = True)
    
    optical_flow.make_joint_overlay_movie(result, 
                                             filename_start + '_joint_result.mp4', 
                                             autoscale = True,
                                             arrow_scale = 0.5,
                                             arrow_boxsize = 4)
 
    plt.figure(figsize = (6.5,4.5),constrained_layout = True)
    plt.subplot(131)
    optical_flow.costum_imshow(result['remodelling'][0,:,:],delta_x = delta_x, autoscale = True, cmap = 'plasma')
    colorbar = plt.colorbar(shrink = 0.6)
    # plt.clim(0.0,0.05)
    plt.title('Inferred remodelling')
    
    plt.subplot(132)
    row = np.linspace(0.0,0.05,movie.shape[2])
    remodelling_matrix = np.tile(row,(movie.shape[1],1))
    optical_flow.costum_imshow(remodelling_matrix,delta_x = delta_x, autoscale = True, cmap = 'plasma')
    # optical_flow.costum_imshow(perturbation,delta_x = delta_x, autoscale = True, cmap = 'plasma')
    plt.ylabel('')
    colorbar = plt.colorbar(shrink = 0.6)
    plt.title('True remodelling')
    # plt.clim(0.0,0.01)

    plt.subplot(133)
    # optical_flow.costum_imshow(result['remodelling'][0,:,:] - perturbation,delta_x = delta_x, autoscale = True, cmap = 'plasma')
    optical_flow.costum_imshow(result['remodelling'][0,:,:] - remodelling_matrix,delta_x = delta_x, autoscale = True, cmap = 'plasma')
    plt.ylabel('')
    colorbar = plt.colorbar(shrink = 0.6)
    plt.title('Difference')
    # plt.clim(0.0,0.05)
    plt.savefig(filename_start + '_remodelling_comparison.pdf')
 
    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
 

if __name__ == '__main__':
    # make_convergence_analysis_simple_example(v_x = 0.1, v_y = 0.2, remodelling = 0.05)
    # make_convergence_analysis_simple_example(v_x = 0.1, v_y = 0.2, remodelling = 0.05, 
                                            #  v_x_start = 0.9, v_y_start = 0.9, remodelling_start = 1.0)

    # try_stopping_condition()
    # illustrate_boundary_artifacts()
    # reproduce_matlab_example_vortex_pair_new()
    analyse_emergence_of_vortex_pair_instability()
    

    ### These are old functions I didn't end up using in my presentation, but didn't want to delete just yet
    # identify_non_uniform_remodelling_rate()
    # reproduce_matlab_variational_flow_method()
    # reproduce_matlab_example_vortex_pair()
    # make_new_test_data_with_non_moving_boundary()
    # apply_method_to_data_with_non_moving_boundary()
    # test_and_time_new_numpy_method()
