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
import matplotlib.ticker
from scipy.stats import gaussian_kde
import cv2
import imageio
import tifffile
import time

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

   
def simle_test_with_data_on_boundary():
    v_x = 0.1
    v_y = 0.2
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 50, include_noise = False)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.5 + v_x, y_position = 2.5 + v_y, sigma = 3, width = 5, dimension = 50, include_noise = False)
    second_frame += 0.05
    movie = np.stack((first_frame, second_frame))
    
    filename_start = os.path.join(os.path.dirname(__file__),'output','simple_example' )
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(filename_start + '_data_.mp4'),dpi=300) 
 
    # call method
    result = optical_flow.variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           speed_alpha=1.0,
                                                           remodelling_alpha = 10000.0,
                                                           smoothing_sigma = None)
    
    optical_flow.make_joint_overlay_movie(result, 
                                             filename_start + '_joint_result.mp4', 
                                             autoscale = True,
                                             arrow_scale = 0.5,
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
 
def test_big_fake_data():
    v_x = 0.1
    v_y = 0.2
    first_frame, delta_x = optical_flow.make_fake_data_frame(x_position = 2.5, y_position = 2.5, sigma = 3, width = 5, dimension = 1000, include_noise = False)
    second_frame, _ = optical_flow.make_fake_data_frame(x_position = 2.5 + v_x, y_position = 2.5 + v_y, sigma = 3, width = 5, dimension = 1000, include_noise = False)
    second_frame += 0.05
    movie = np.stack((first_frame, second_frame))
    
    filename_start = os.path.join(os.path.dirname(__file__),'output','big_file_example')
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(filename_start + '_data_.mp4'),dpi=300) 
 
    # call method
    result = optical_flow.variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = 1.0,
                                                           speed_alpha=1e8,
                                                           remodelling_alpha = 4*1e8)
    
    optical_flow.make_joint_overlay_movie(result, 
                                             filename_start + '_joint_result.mp4', 
                                             autoscale = True,
                                             arrow_scale = 0.5,
                                             arrow_boxsize = 80)
 
    # formatter = matplotlib.ticker.ScalarFormatter()
    # formatter.set_powerlimits((-2, 2))
    # plt.gca().images[-1].colorbar.ax.yaxis.set_major_formatter(formatter)
    # plt.gca().images[-1].colorbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    print('mean and max final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print('mean and max final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print('mean and max final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
 
def reproduce_matlab_example_vortex_pair(speed_regularisation=2e4, remodelling_regularisation=1e3):
    first_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_1.tif')) 
    second_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_2.tif')) 

    print("the mean intensity difference of the two frames is")
    print(np.mean(np.abs(first_frame - second_frame)))
    print("the difference of mean intensities of the two frames is")
    print(np.mean(first_frame) - np.mean(second_frame))
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
    result = optical_flow.variational_optical_flow(movie,
                                                           delta_x = 1.0,
                                                           delta_t = 1.0,
                                                           speed_alpha=speed_regularisation,
                                                           remodelling_alpha=remodelling_regularisation,
                                                           initial_v_x =0.015,
                                                           initial_v_y =0.015,
                                                        #    remodelling_guess=0.05,
    # )
                                                           smoothing_sigma = 0.62*4)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'vortex_test_' + str(speed_regularisation) + ' '
                                                          + str(remodelling_regularisation) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.1,
                                             arrow_boxsize = 20,
                                             arrow_width = 0.005)
                                            #  arrow_color = 'lime')

    optical_flow.make_joint_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                             'vortex_test' + str(speed_regularisation) + ' '
                                                          + str(remodelling_regularisation) + '_joint_result.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.1,
                                             arrow_boxsize = 20,
                                             arrow_width = 0.005)

    print('mean, max and min final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print(np.min(result['v_x']))

    print('mean, max and min final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print(np.min(result['v_y']))

    print('mean, max and min final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
    print(np.min(result['remodelling']))
 
def perform_tuning_variation_on_vortex_example():
    first_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_1.tif')) 
    second_frame = tifffile.imread(os.path.join(os.path.dirname(__file__),'data','vortex_pair_particles_2.tif')) 
    movie = np.stack((first_frame, second_frame))
    
    # result_for_plotting = optical_flow.vary_regularisation(movie, speed_alpha_values = np.linspace(1000,100000,15),
                                                        #    remodelling_alpha_values = np.linspace(100,10000,15),
                                                        #    filename = os.path.join(os.path.dirname(__file__), 'output',
                                                                                #    'vortex_pair_regularisation_variation'),
                                                        #    smoothing_sigma = 0.62*4)

    result_for_plotting = np.load(os.path.join(os.path.dirname(__file__),'output','vortex_pair_regularisation_variation.npy'),allow_pickle='TRUE').item()
    optical_flow.plot_regularisation_variation(result_for_plotting, os.path.join(os.path.dirname(__file__), 'output',
                                                                                   'vortex_pair_regularisation_variation.pdf'))
    
    print(np.min(result_for_plotting["converged"]))
    print(np.max(result_for_plotting["converged"]))
    print(np.sum(result_for_plotting["converged"]))

    
def apply_to_bischoff_data(speed_regularisation=6000, remodelling_regularisation=6000):

    movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','MB301110_i_4_movie_8 bit.tif'))
    delta_x = 105/1024
    delta_t = 10

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    # ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    # ani.save(os.path.join(os.path.dirname(__file__),'output','pretty_real_movie.mp4'),dpi=300) 
 
    # iterations = 10000
    # iteration_stepsize = 500
    movie = movie[3:5,:,:]
    # movie = movie[:10,:,:]
    result = optical_flow.variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = delta_t,
                                                           speed_alpha=speed_regularisation,
                                                           remodelling_alpha=remodelling_regularisation,
                                                        #    initial_v_x =0.015,
                                                        #    initial_v_y =0.015,
                                                        #    remodelling_guess=0.05,
    # )
                                                           smoothing_sigma = 5)
    
    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'real_data_test_' + str(speed_regularisation) + ' '
                                                          + str(remodelling_regularisation) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.5,
                                             arrow_boxsize = 10)
                                            #  arrow_width = 0.005)
                                            #  arrow_color = 'lime')

    optical_flow.make_joint_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                             'real_data_test' + str(speed_regularisation) + ' '
                                                          + str(remodelling_regularisation) + '_joint_result.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.5,
                                             arrow_boxsize = 10)
                                            #  arrow_width = 0.005)

    print('mean, max and min final v_x are')
    print(np.mean(result['v_x']))
    print(np.max(result['v_x']))
    print(np.min(result['v_x']))

    print('mean, max and min final v_y are')
    print(np.mean(result['v_y']))
    print(np.max(result['v_y']))
    print(np.min(result['v_y']))

    print('mean, max and min final remodelling are')
    print(np.mean(result['remodelling']))
    print(np.max(result['remodelling']))
    print(np.min(result['remodelling']))
 
def perform_tuning_variation_on_real_data():
    movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','MB301110_i_4_movie_8 bit.tif'))
    delta_x = 105/1024
    delta_t = 10

    # fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    # def animate(i): 
        # plt.cla()
        # optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    # ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    # ani.save(os.path.join(os.path.dirname(__file__),'output','pretty_real_movie.mp4'),dpi=300) 
 
    # iterations = 10000
    # iteration_stepsize = 500
    # movie = movie[3:5,:,:]
    movie = movie[3:5,:,:]
 
    result_for_plotting = optical_flow.vary_regularisation(movie, speed_alpha_values = np.logspace(3,8,15),
                                                           remodelling_alpha_values = np.logspace(-1,8,20),
                                                           filename = os.path.join(os.path.dirname(__file__), 'output',
                                                                                   'real_data_regularisation_variation'),
                                                           smoothing_sigma = 5)

    # result_for_plotting = np.load(os.path.join(os.path.dirname(__file__),'output','real_data_regularisation_variation.npy'),allow_pickle='TRUE').item()
    optical_flow.plot_regularisation_variation(result_for_plotting, os.path.join(os.path.dirname(__file__), 'output',
                                                                                   'real_data_regularisation_variation.pdf'),
                                                                                   use_log_axes = True)
    

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
    # simle_test_with_data_on_boundary()
    # test_big_fake_data()
    # reproduce_matlab_example_vortex_pair(speed_regularisation=2e4, remodelling_regularisation=1e3)
    # reproduce_matlab_example_vortex_pair(speed_regularisation=1e5, remodelling_regularisation=1e4)
    # perform_tuning_variation_on_vortex_example()

    # apply_to_bischoff_data()
    perform_tuning_variation_on_real_data()
    # try_stopping_condition()
    # test_with_data_on_boundary()
    

    ### These are old functions I didn't end up using in my presentation, but didn't want to delete just yet
    # identify_non_uniform_remodelling_rate()
    # reproduce_matlab_variational_flow_method()
    # reproduce_matlab_example_vortex_pair()
    # make_new_test_data_with_non_moving_boundary()
    # apply_method_to_data_with_non_moving_boundary()
    # test_and_time_new_numpy_method()
