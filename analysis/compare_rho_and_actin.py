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

delta_x = 0.0913
delta_t = 10.0

def make_joint_movie(smoothing_sigma = None, use_clahe = False):
    """make a movie with both color channels together. If a sigma is provided, it will use that to blur the movie first"""
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))

       
    if use_clahe:
        rho_movie = optical_flow.apply_clahe(rho_movie, clipLimit = 10000)
        actin_movie = optical_flow.apply_clahe(actin_movie, clipLimit = 10000)
        clahe_string = '_w_clahe_'
    else:
        clahe_string = ''

    if smoothing_sigma is not None:
        rho_movie = optical_flow.blur_movie(rho_movie, smoothing_sigma)
        actin_movie = optical_flow.blur_movie(actin_movie, smoothing_sigma)
        smoothing_string = '_sigma_' + "{:.2f}".format(smoothing_sigma)
    else:
        smoothing_string = ''
    
 
    fig = plt.figure(figsize = (4.5,2.5))
    def animate(i): 
        # plt.cla()
        plt.subplot(121)
        # plt.gca().set_axis_off()
        plt.title('Rho')
        optical_flow.costum_imshow(rho_movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(rho_movie))
        plt.subplot(122)
        plt.title('Actin')
        optical_flow.costum_imshow(actin_movie[i,:,:],delta_x = delta_x, v_max = np.max(actin_movie))
        # plt.imshow(actin_movie[i,:,:],cmap = 'gray_r',vmin = 0, vmax = 255, interpolation = None)
        # plt.gca().set_axis_off()
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
        # plt.savefig(os.path.join(os.path.dirname(__file__),'output','joint_movie' + str(i) + '.png'),dpi=300) 
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','joint_movie' + smoothing_string + clahe_string + '.mp4'),dpi=300) 
    
def make_actin_clahe_movie(smoothing_sigma = None):
    """make a movie with both color channels together. If a sigma is provided, it will use that to blur the movie first"""
    
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))

    clahed_actin_movie = optical_flow.apply_clahe(actin_movie, clipLimit = 50000, tile_number = 10)
    
    if smoothing_sigma is not None:
        actin_movie = optical_flow.blur_movie(actin_movie, smoothing_sigma)
        clahed_actin_movie = optical_flow.blur_movie(clahed_actin_movie, smoothing_sigma)
        smoothing_string = '_sigma_' + "{:.2f}".format(smoothing_sigma)
    else:
        smoothing_string = ''
 
    fig = plt.figure(figsize = (4.5,2.5))
    def animate(i): 
        # plt.cla()
        plt.subplot(121)
        # plt.gca().set_axis_off()
        plt.title('without CLAHE')
        optical_flow.costum_imshow(actin_movie[i,:,:],delta_x = delta_x)
        plt.subplot(122)
        plt.title('with CLAHE')
        optical_flow.costum_imshow(clahed_actin_movie[i,:,:],delta_x = delta_x, v_min = 0.0, v_max = np.max(clahed_actin_movie))
        # plt.imshow(actin_movie[i,:,:],cmap = 'gray_r',vmin = 0, vmax = 255, interpolation = None)
        # plt.gca().set_axis_off()
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
        # plt.savefig(os.path.join(os.path.dirname(__file__),'output','joint_movie' + str(i) + '.png'),dpi=300) 
    ani = FuncAnimation(fig, animate, frames=actin_movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','actin_clahe_movie' + smoothing_string + '.mp4'),dpi=300) 
 
def investigate_intensities():
    """This makes a figure with the intensity histograms of both movies"""
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    
    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.subplot(121)
    plt.hist(actin_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Actin intensity value')
    plt.xlim(-2,120)
    plt.ylabel('Number of pixels')
    
    plt.subplot(122)
    plt.hist(rho_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Rho intensity value')
    plt.ylabel('Number of pixels')
    plt.xlim(-2,120)
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','both_intensity_histgrams.pdf'))
    
    print('Unique intensity values are')
    print(np.unique(actin_movie))
 
def make_blurring_analysis(channel = 'actin'):
    """This makes a movie to analise how the blurring affects the histogram and spatial intensity profile of one frame"""

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    this_frame = movie[3,:,:]
    this_frame_rgb = np.zeros((this_frame.shape[0], this_frame.shape[1], 3),dtype = 'int')
    this_frame_rgb[:,:,0] = 255 - this_frame
    this_frame_rgb[:,:,1] = 255 - this_frame
    this_frame_rgb[:,:,2] = 255 - this_frame
    
    Xpixels=this_frame.shape[0]#Number of X pixels
    Ypixels=this_frame.shape[1]#Number of Y pixels
    x_extent = Xpixels * delta_x
    y_extent = Ypixels * delta_x

    x_position = 12.0
    x_index_for_line = round(x_position/x_extent*this_frame.shape[0])
    highlighted_line = this_frame[x_index_for_line,:]
    this_frame_rgb[x_index_for_line,:,0] = 0
    this_frame_rgb[x_index_for_line,:,2] = 0
    y_positions_on_line = np.arange(0,this_frame.shape[1],1)*delta_x

    original_histogram, bin_edges = np.histogram(this_frame.flatten(), range = (0,255), bins=255)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    original_histogram = original_histogram/np.max(original_histogram)

    # blur_sizes = np.arange(0.1,5,0.01)
    blur_sizes = np.arange(0.1,5,0.01)
    # blur_sizes = np.arange(0.1,5,2.5)
    
    fig = plt.figure(figsize = (4.5,4.5), constrained_layout = True)
    def animate_blurr(index):
        plt.clf()

        outer_grid = fig.add_gridspec(2,2,hspace = 0.1)
        left_grid = outer_grid[1,0].subgridspec(2,1)
        fig.suptitle('Blur with $\mathrm{\sigma}$=' + "{:.2f}".format(blur_sizes[index]) )

        this_filtered_image = skimage.filters.gaussian(this_frame, sigma =blur_sizes[index], preserve_range = True)
        this_histogram, _ = np.histogram(this_filtered_image.flatten(), range = (0,255), bins=255)
        this_histogram = this_histogram.astype('float')
        this_histogram/= np.max(this_histogram)
        this_filtered_line = this_filtered_image[x_index_for_line]

        ax1 = fig.add_subplot(outer_grid[0,0], anchor = (0.75,1.0))
        ax1.imshow(this_frame_rgb, extent = [0,y_extent, x_extent, 0], interpolation = None )
        ax1.set_xlabel("y-position [$\mathrm{\mu}$m]")
        ax1.set_ylabel("x-position [$\mathrm{\mu}$m]")

        ax2 = fig.add_subplot(outer_grid[0,1])
        optical_flow.costum_imshow(this_filtered_image, delta_x = delta_x)

        ax3 = fig.add_subplot(left_grid[0,0])
        ax3.step(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.6, linewidth = 0.15)
        ax3.step(y_positions_on_line,this_filtered_line, linewidth =0.2, where = 'mid')
        ax3.set_ylabel('Intensity')

        ax4 = fig.add_subplot(left_grid[1,0])
        ax4.step(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.6, linewidth = 0.15, where = 'mid' )
        ax4.step(y_positions_on_line,this_filtered_line, linewidth =0.2, where = 'mid')
        ax4.set_xlabel('y-position [$\mathrm{\mu}$m]')
        ax4.set_ylabel('Intensity')
        ax4.set_xlim(5,15)

        ax5 = fig.add_subplot(outer_grid[1,1], anchor = (0.5,4.5), box_aspect = 1.1)
        ax5.bar(bin_centers, this_histogram, width = 1.0)
        ax5.bar(bin_centers, original_histogram, width = 1.0, color = 'grey', alpha = 0.8)
        ax5.set_xlim(0,120)
        ax5.set_xlabel('Image intensity')
        ax5.set_ylabel('Normalised occupancy')

    animation = FuncAnimation(fig, animate_blurr, frames=len(blur_sizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','blur_analysis_' + channel + '.mp4'),dpi=300) 
 
def investigate_intensity_thresholds():
    """This function plots both histograms again, but using blurred data. It identifies the intensity value separating the two modes of the histogram"""
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    
    blurred_actin_movie = optical_flow.blur_movie(actin_movie, smoothing_sigma=1.3)
    blurred_rho_movie = optical_flow.blur_movie(rho_movie, smoothing_sigma=1.0)


    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.subplot(121)
    plt.hist(blurred_actin_movie.flatten(),bins=255, range = (0,255))
    plt.xlim(0,100)
    plt.axvline(17, color = 'black', label = 'Intensity = 17')
    plt.xlabel('Actin intensity value')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.gca().ticklabel_format(scilimits = (-3,3))
    
    plt.subplot(122)
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    plt.hist(blurred_rho_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Rho intensity value')
    plt.ylabel('Number of pixels')
    plt.xlim(0,100)
    plt.axvline(18, color = 'black', label = 'Intensity = 18')
    plt.legend()
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','both_intensity_histgrams_blurred.pdf'))
 
def make_thresholded_movies( threshold = 17.5, rho_sigma = 1.0, actin_sigma = 1.3, clahe = None, adaptive = False):
    """This function makes a movie with both channels, and in which pixel values below a certain intensity are coloured in green."""
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    
    if clahe is not None:
        rho_movie = optical_flow.apply_clahe(rho_movie, clipLimit = clahe)
        actin_movie = optical_flow.apply_clahe(actin_movie, clipLimit = clahe)
        rho_movie /= np.max(rho_movie)
        rho_movie*=255.0
        actin_movie /= np.max(actin_movie)
        actin_movie*=255.0
        clahe_string = '_w_clahe_'
    else:
        clahe_string = ''
    
    rho_movie_blurred = optical_flow.blur_movie(rho_movie, smoothing_sigma = rho_sigma)
    actin_movie_blurred = optical_flow.blur_movie(actin_movie, smoothing_sigma = actin_sigma)
    
    if adaptive:
        rho_mask = optical_flow.apply_adaptive_threshold(rho_movie_blurred, window_size = 151, threshold = -5)
        actin_mask = optical_flow.apply_adaptive_threshold(actin_movie_blurred, window_size = 151, threshold = -5)
    else:
        rho_mask = rho_movie<threshold
        actin_mask = actin_movie<threshold
    
    rho_thresholded = np.zeros((actin_movie.shape[0],actin_movie.shape[1],actin_movie.shape[2], 3), dtype = 'int')
    actin_thresholded = np.zeros((actin_movie.shape[0],actin_movie.shape[1],actin_movie.shape[2], 3), dtype = 'int')
    
    actin_thresholded[np.logical_not(actin_mask),1] = 255-actin_movie[np.logical_not(actin_mask)]
    actin_thresholded[actin_mask,0] = 255-actin_movie[actin_mask]
    actin_thresholded[actin_mask,1] = 255-actin_movie[actin_mask]
    actin_thresholded[actin_mask,2] = 255-actin_movie[actin_mask]
    
    rho_thresholded[np.logical_not(rho_mask),1] = 255-rho_movie[np.logical_not(rho_mask)]
    rho_thresholded[rho_mask,0] = 255-rho_movie[rho_mask]
    rho_thresholded[rho_mask,1] = 255-rho_movie[rho_mask]
    rho_thresholded[rho_mask,2] = 255-rho_movie[rho_mask]

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        plt.subplot(121)
        # plt.gca().set_axis_off()
        plt.title('Rho')
        optical_flow.costum_imshow(rho_thresholded[i,:,:,:],autoscale = False, cmap = None, delta_x = delta_x)
        plt.subplot(122)
        plt.title('Actin')
        optical_flow.costum_imshow(actin_thresholded[i,:,:,:],autoscale = False, cmap = None, delta_x = delta_x)
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0])
    ani.save(os.path.join(os.path.dirname(__file__),'output','joint_movie_thresholded_treshold_' + "{:.2f}".format(threshold) + 
                          '_rho_' + "{:.2f}".format(rho_sigma)+ '_actin_' +  "{:.2f}".format(actin_sigma) + clahe_string + '.mp4'),dpi=300) 
 
    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.subplot(121)
    plt.hist(actin_movie_blurred.flatten(),bins=255, range = (0,255))
    # plt.xlim(0,100)
    plt.axvline(17, color = 'black', label = 'Intensity = 17')
    plt.xlabel('Actin intensity value')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.gca().ticklabel_format(scilimits = (-3,3))
    
    plt.subplot(122)
    plt.hist(rho_movie_blurred.flatten(),bins=255, range = (0,255))
    plt.xlabel('Rho intensity value')
    plt.ylabel('Number of pixels')
    # plt.xlim(0,100)
    plt.axvline(18, color = 'black', label = 'Intensity = 18')
    plt.legend()
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','both_intensity_histgrams_blurred_clahed.pdf'))
 

def check_error_of_method(include_noise = False):
    """This makes a movie of with silico data and applies optical flow. It then saves a visualisation of the result
    as a movie and makes some histograms."""
    
    x_velocity = 0.1
    y_velocity = 0.2
    delta_t = 0.5
    
    if include_noise:
        file_name_addition = '_with_noise'
    else:
        file_name_addition = '_without_noise'

    x_step = x_velocity*delta_t
    y_step = y_velocity*delta_t  
    n_steps = 5
    frames = []
    x_position = 5
    y_position = 3
    for frame_index in range(n_steps):
        this_frame, delta_x = optical_flow.make_fake_data_frame(x_position, y_position, include_noise)
        frames.append(this_frame)
        x_position += x_step
        y_position += y_step
        
    fake_data = np.array(frames)

    this_result = optical_flow.conduct_optical_flow(fake_data, boxsize = 15, delta_x = delta_x, delta_t = 0.5)
    np.save(os.path.join(os.path.dirname(__file__),'output','fake_flow_result' + file_name_addition + '.npy'), this_result)
    
    this_result['original_data'][:-1,:,:][this_result['v_x'] == np.inf] = 1.0
    this_result['original_data'][:-1,:,:][this_result['v_y'] == np.inf] = 1.0
    this_result['v_x'][this_result['v_x'] == np.inf] = 0.0
    print('max vx is')
    print(np.nanmax(this_result['v_x']))
    print('min vx is')
    print(np.nanmin(this_result['v_x']))
    print('mean vx is')
    print(np.nanmean(this_result['v_x']))

    this_result['v_y'][this_result['v_y'] == np.inf] = 0.0
    print('max vy is')
    print(np.nanmax(this_result['v_y']))
    print('min vy is')
    print(np.nanmin(this_result['v_y']))
    print('mean vy is')
    print(np.nanmean(this_result['v_y']))
    
    print('true vx is ' + str(x_velocity))
    print('true vy is ' + str(y_velocity))
 

    plt.figure(figsize = (4.5,2.5))
    plt.subplot(121)
    plt.hist(this_result['v_x'].flatten(), bins=100, density=False)
    plt.xlabel('$\mathrm{v_x}$ values')
    plt.ylabel('Number of Pixels')
    plt.axvline(x_velocity, color = 'red', lw =0.2)
    plt.gca().ticklabel_format(scilimits = (-3,3))

    plt.subplot(122)
    plt.hist(this_result['v_y'].flatten(), bins=100, density=False)
    plt.xlabel('$\mathrm{v_y}$ values')
    plt.ylabel('Number of Pixels')
    plt.axvline(y_velocity, color = 'red', lw = 0.2)
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.tight_layout() 
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','fake_v_histogram' + file_name_addition + '.pdf')) 
    
    optical_flow.make_velocity_overlay_movie(this_result, 
                                             os.path.join(os.path.dirname(__file__),'output','made_up_data_velocities' + file_name_addition + '.mp4'),
                                             arrow_boxsize = 40,
                                             arrow_scale = 0.2,
                                             autoscale = True)

def make_boxsize_analysis(channel = 'actin', blursize = 1.3):
    """This function analyses how optical flow results depend on the boxsize. Makes a multi-panal movie and a number of additional figures."""

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
        smoothing_sigma = 1.0
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
        smoothing_sigma = 1.3
    
    boxsizes = np.arange(5,150,2).astype('int')
    # boxsizes = np.arange(5,90,4).astype('int')
    # boxsizes = np.arange(5,56,50).astype('int')
    
    test_locations = np.array([[12.5,7],
                               [20,15],
                               [22,19],
                               [30,19]])
    
    location_indices = np.zeros_like(test_locations, dtype = 'int')
    location_indices[:,0] = test_locations[:,0]/(movie.shape[1]*delta_x)*movie.shape[1]
    location_indices[:,1] = test_locations[:,1]/(movie.shape[2]*delta_x)*movie.shape[2]

    mean_velocities = np.zeros_like(boxsizes, dtype = 'float')
    velocities_std = np.zeros_like(boxsizes, dtype = 'float')
    local_velocities = np.zeros((test_locations.shape[0],len(boxsizes)), dtype = 'float')

    fig = plt.figure(figsize = (4.5,4.5), constrained_layout = True)
    def animate_boxsizes(index):
        plt.clf()
        boxsize = boxsizes[index]
        boxsize_in_micro_meters = boxsize*delta_x
        fig.suptitle('Boxsize b=' + str(boxsize) + r' ($\approx$' + "{:.2f}".format(boxsize_in_micro_meters) + '$\mathrm{\mu}$m)' )
        this_result = optical_flow.conduct_optical_flow(movie[3:5,:,:], boxsize = boxsizes[index], delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = blursize)
        cos_values = this_result['v_y']/this_result['speed']
        angles = np.arccos(cos_values)*np.sign(this_result['v_x'])
        x_positions, y_positions, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(this_result, arrow_boxsize = 15)
        this_mean_speed = np.mean(this_result['speed'])
        this_speed_std = np.std(this_result['speed'])
        mean_velocities[index] = this_mean_speed
        velocities_std[index] = this_speed_std
        for position_index, location_index_pair in enumerate(location_indices):
            local_velocities[position_index, index] = this_result['speed'][0,location_index_pair[0],location_index_pair[1]]

        outer_grid = fig.add_gridspec(2,2)
        right_grid = outer_grid[1,1].subgridspec(2,1)

        ax1 = fig.add_subplot(outer_grid[0,0])
        optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
        ax1.quiver(y_positions, x_positions, v_y[0,:,:], -v_x[0,:,:], color = 'magenta',headwidth=5, scale = None)

        ax2 = fig.add_subplot(outer_grid[0,1])
        optical_flow.costum_imshow(this_result['speed'][0,:,:],autoscale = True, cmap = None, delta_x = delta_x)
        colorbar = plt.colorbar()
        plt.clim(0,0.1)
        colorbar.ax.set_ylabel('Motion speed [$\mathrm{\mu m}$/s]')

        ax3 = fig.add_subplot(outer_grid[1,0])
        plt.hist(this_result['speed'].flatten(), bins = 50, density = False)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.xlabel('Motion speed [$\mathrm{\mu m}$/s]')
        # plt.xlim((0,0.02))
        plt.ylabel('Number of Pixels')
        
        ax4 = fig.add_subplot(right_grid[0,0])
        plt.hist(angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        plt.xlabel('Angle to y axis')
        plt.ylabel('# Pixels')
        
        ax5 = fig.add_subplot(right_grid[1,0])
        plt.hist(angles.flatten()/np.pi, bins = 50, density = False, weights = this_result['speed'].flatten())
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        plt.xlabel('Weighted angle to y axis')
        plt.ylabel('# Pixels')
        
    animation = FuncAnimation(fig, animate_boxsizes, frames=len(boxsizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','boxsize_analysis_' + channel + '_blursize_' + str(blursize) +  '.mp4'),dpi=300) 

    plt.figure(figsize = (4.5,2.5))
    plt.subplot(121)
    plt.plot(boxsizes, mean_velocities)
    plt.xlabel('boxsize')
    plt.ylabel('mean speed [$\mathrm{\mu m}$/s]')
    plt.subplot(122)
    plt.plot(boxsizes, velocities_std)
    plt.xlabel('boxsize')
    plt.ylabel('speed standard dev. [$\mathrm{\mu m}$/s]')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','new_boxsize_velocities_' + channel  + '_blursize_' + str(blursize) + '.pdf')) 
 
    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.suptitle('Blursize ' + "{:.2f}".format(blursize))
    plt.subplot(121)
    optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
    for test_location in test_locations:
        plt.scatter( test_location[1], test_location[0], marker = 'x')
    plt.subplot(122)
    for recorded_values in local_velocities:
        plt.plot(boxsizes, recorded_values)
    plt.xlabel('boxsize')
    plt.ylabel('Local speed [$\mathrm{\mu m}$/s]')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','boxsize_local_velocities_' + channel + '_blursize_' + str(blursize) + '.pdf'),dpi = 300) 

def make_OF_blur_analysis(channel = 'actin', boxsize = 21):
    """This function analyses how optical flow results depend on the boxsize. Makes a multi-panal movie and a number of additional figures."""

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    # blur_sizes = np.arange(0.1,5,0.01)
    # blur_sizes = np.arange(0.1,5.0,0.01)
    blur_sizes = np.arange(0.5,15,0.1)
    # blur_sizes = np.arange(0.5,5,3.6)
    
    mean_velocities = np.zeros_like(blur_sizes, dtype = 'float')
    velocities_std = np.zeros_like(blur_sizes, dtype = 'float')

    test_locations = np.array([[12.5,7],
                               [20,15],
                               [22,19],
                               [30,19]])
    
    location_indices = np.zeros_like(test_locations, dtype = 'int')
    location_indices[:,0] = test_locations[:,0]/(movie.shape[1]*delta_x)*movie.shape[1]
    location_indices[:,1] = test_locations[:,1]/(movie.shape[2]*delta_x)*movie.shape[2]

    local_velocities = np.zeros((test_locations.shape[0],len(blur_sizes)), dtype = 'float')


    x_loc = 12.5
    y_loc = 7
    x_index = round(x_loc/(movie.shape[1]*delta_x)*movie.shape[1])
    y_index = round(y_loc/(movie.shape[2]*delta_x)*movie.shape[2])

    fig = plt.figure(figsize = (4.5,4.5), constrained_layout = True)
    def animate_blursizes(index):
        plt.clf()
        blursize = blur_sizes[index]
        blursize_in_micro_meters = blursize*delta_x
        fig.suptitle('Blur $\mathrm{\sigma}$=' + "{:.2f}".format(blursize) + r' ($\approx$' + "{:.2f}".format(blursize_in_micro_meters) + '$\mathrm{\mu}$m)' )
        this_result = optical_flow.conduct_optical_flow(movie[3:5,:,:], boxsize = boxsize, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = blursize)
        cos_values = this_result['v_y']/this_result['speed']
        angles = np.arccos(cos_values)*np.sign(this_result['v_x'])
        x_positions, y_positions, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(this_result, arrow_boxsize = 15)
        this_mean_speed = np.mean(this_result['speed'])
        this_speed_std = np.std(this_result['speed'])
        mean_velocities[index] = this_mean_speed
        velocities_std[index] = this_speed_std
        for position_index, location_index_pair in enumerate(location_indices):
            local_velocities[position_index, index] = this_result['speed'][0,location_index_pair[0],location_index_pair[1]]

        outer_grid = fig.add_gridspec(2,1)
        upper_grid = outer_grid[0,0].subgridspec(1,3,wspace = 0.0)
        lower_grid = outer_grid[1,0].subgridspec(1,2)
        right_grid = lower_grid[0,1].subgridspec(2,1)

        ax1 = fig.add_subplot(upper_grid[0,0])
        ax1.set_title('blur hidden')
        optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
        ax1.quiver(y_positions, x_positions, v_y[0,:,:], -v_x[0,:,:], color = 'magenta',headwidth=5, scale = None)

        ax2 = fig.add_subplot(upper_grid[0,1])
        ax2.set_title('blur visible')
        optical_flow.costum_imshow(this_result['blurred_data'][1,:,:], delta_x = delta_x)
        ax2.get_yaxis().set_visible(False)
        ax2.set_ylabel('')
        ax2.quiver(y_positions, x_positions, v_y[0,:,:], -v_x[0,:,:], color = 'magenta',headwidth=5, scale = None)

        ax3 = fig.add_subplot(upper_grid[0,2])
        ax3.set_title('speed')
        optical_flow.costum_imshow(this_result['speed'][0,:,:],autoscale = True, cmap = None, delta_x = delta_x)
        ax3.get_yaxis().set_visible(False)
        colorbar = plt.colorbar()
        colorbar.ax.set_ylabel('Motion speed [$\mathrm{\mu m}$/s]')
        plt.clim(0,0.3)
        # plt.clim(0,0.06)
        ax3.set_ylabel('')


        ax4 = fig.add_subplot(lower_grid[0,0])
        ax4.hist(this_result['speed'].flatten(), bins = 50, density = False)
        ax4.ticklabel_format(scilimits = (-3,3))
        ax4.set_xlabel('Motion speed [$\mathrm{\mu m}$/s]')
        # plt.xlim((0,0.02))
        ax4.set_ylabel('Number of Pixels')
        
        ax5 = fig.add_subplot(right_grid[0,0])
        ax5.hist(angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
        ax5.ticklabel_format(scilimits = (-3,3))
        # ax5.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        # ax5.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        ax5.set_xlabel('Angle to y axis')
        ax5.set_ylabel('# Pixels')
        
        ax6 = fig.add_subplot(right_grid[1,0], sharex = ax5)
        ax6.hist(angles.flatten()/np.pi, bins = 50, density = False, weights = this_result['speed'].flatten())
        ax6.ticklabel_format(scilimits = (-3,3))
        ax6.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        ax6.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        ax6.set_xlabel('Weighted angle to y axis')
        ax6.set_ylabel('# Pixels')
        
    animation = FuncAnimation(fig, animate_blursizes, frames=len(blur_sizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','blursize_analysis_' + channel + '_boxsize_' + str(boxsize) + '.mp4'),dpi=300) 

    plt.figure(figsize = (4.5,2.5))
    plt.subplot(121)
    plt.plot(blur_sizes, mean_velocities)
    plt.xlabel('blursize')
    plt.ylabel('mean speed [$\mathrm{\mu m}$/s]')
    plt.subplot(122)
    plt.plot(blur_sizes, velocities_std)
    plt.xlabel('blursize')
    plt.ylabel('speed standard dev. [$\mathrm{\mu m}$/s]')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','blursize_velocities_' + channel + '_boxsize_' + str(boxsize) + '.pdf')) 
 
    movie[4,x_index,:] = 255
    movie[4,:,y_index] = 255
    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.suptitle('Boxsize ' + str(boxsize))
    plt.subplot(121)
    optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
    for test_location in test_locations:
        plt.scatter( test_location[1], test_location[0], marker = 'x')
    plt.subplot(122)
    for recorded_values in local_velocities:
        plt.plot(blur_sizes, recorded_values)
    plt.xlabel('blursize')
    plt.ylabel('Local speed [$\mathrm{\mu m}$/s]')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','blursize_local_velocities_' + channel + '_boxsize_' + str(boxsize) + '.pdf'),dpi = 300) 

def make_and_save_rho_optical_flow(include_remodelling = False):
    """This function saves the optical flow result for rho"""
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(rho_movie, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = 3.0, boxsize = 31, include_remodelling = include_remodelling)
    
    if include_remodelling:
        remodelling_string = '_w_remodelling'
    else:
        remodelling_string = ''
    
    np.save(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result' + remodelling_string + '.npy'), this_result)

def make_and_save_actin_optical_flow(include_remodelling = False):
    """This function saves the optical flow result for actin"""
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(actin_movie, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = 3.0, boxsize = 31, include_remodelling = include_remodelling)

    if include_remodelling:
        remodelling_string = '_w_remodelling'
    else:
        remodelling_string = ''
    
    np.save(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result' + remodelling_string + '.npy'), this_result)

def joint_actin_and_rho_flow_result(show_blurred=False, include_remodelling = False):
    """This function shows the optical flow result of channels in one movie, using the saved files from the functions above"""

    if include_remodelling:
        remodelling_string = '_w_remodelling'
    else:
        remodelling_string = ''

    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result' + remodelling_string + '.npy'),allow_pickle='TRUE').item()
    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result' + remodelling_string + '.npy'),allow_pickle='TRUE').item()

    x_positions, y_positions, v_x_rho, v_y_rho = optical_flow.subsample_velocities_for_visualisation(rho_flow_result, arrow_boxsize = 15)
    x_positions, y_positions, v_x_actin, v_y_actin = optical_flow.subsample_velocities_for_visualisation(actin_flow_result, arrow_boxsize = 15)

    
    if show_blurred:
        rho_movie = rho_flow_result['blurred_data']
        actin_movie = actin_flow_result['blurred_data']
        filename = 'joint_overlay_blurred' + remodelling_string + '.mp4'
    else:
        rho_movie = rho_flow_result['original_data']
        actin_movie = actin_flow_result['original_data']
        filename = 'joint_overlay' + remodelling_string + '.mp4'

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(121)
        plt.title('Rho')
        optical_flow.costum_imshow(rho_movie[i,:,:],delta_x = delta_x)
        plt.quiver(y_positions, x_positions, v_y_rho[i,:,:], -v_x_rho[0,:,:], color = 'magenta',headwidth=5, scale = None)
        plt.subplot(122)
        plt.title('Actin')
        optical_flow.costum_imshow(actin_movie[i,:,:],delta_x = delta_x)
        plt.quiver(y_positions, x_positions, v_y_actin[i,:,:], -v_x_actin[0,:,:], color = 'magenta',headwidth=5, scale = None)
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0]-1)

    ani.save(os.path.join(os.path.dirname(__file__),'output',filename),dpi=300) 

def visualise_actin_and_rho_velocities():
    """This function makes a velocity overlay from the saved results, but with one file for each channel"""
    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(actin_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','actin_velocities.mp4'),arrow_scale = 1.0, arrow_boxsize = 15)

    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(rho_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','rho_velocities.mp4'), arrow_boxsize = 15)
 
def make_joint_speed_and_angle_histograms():
    """This function analyses how the velocities of both channels correspond. It makes multiple figures."""
    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'),allow_pickle='TRUE').item()
    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'),allow_pickle='TRUE').item()
    
    rho_cos_values = rho_flow_result['v_y']/rho_flow_result['speed']
    rho_angles = np.arccos(rho_cos_values)*np.sign(rho_flow_result['v_x'])

    actin_cos_values = actin_flow_result['v_y']/actin_flow_result['speed']
    actin_angles = np.arccos(actin_cos_values)*np.sign(actin_flow_result['v_x'])

    plt.figure(figsize = (4.5,4.5), constrained_layout = True)

    plt.subplot(221)
    plt.hist(actin_flow_result['speed'].flatten(), bins = 50, range = (0,0.1))
    plt.ylabel('Number of pixels')
    plt.xlabel('Speed [$\mathrm{\mu m}$/s]')
    plt.title('Actin')
    plt.gca().ticklabel_format(scilimits = (-3,3))

    plt.subplot(222)
    plt.title('Rho')
    plt.hist(rho_flow_result['speed'].flatten(), bins = 50, range = (0,0.1))
    plt.ylabel('Number of pixels')
    plt.xlabel('Speed [$\mathrm{\mu m}$/s]')
    plt.gca().ticklabel_format(scilimits = (-3,3))
    
    plt.subplot(223)
    plt.hist(actin_angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    plt.xlabel('Angle to y axis')
    plt.ylabel('Number of pixels')
 
    plt.subplot(224)
    plt.hist(rho_angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    plt.xlabel('Angle to y axis')
    plt.ylabel('Number of pixels')

    plt.savefig(os.path.join(os.path.dirname(__file__),'output','joint_speed_histograms.pdf'),dpi=300) 
    
    scalar_products = (actin_flow_result['v_x']*rho_flow_result['v_x'] +
                       actin_flow_result['v_y']*rho_flow_result['v_y'])
    
    cos_values = scalar_products/(actin_flow_result['speed']*rho_flow_result['speed'])
    theta_values = np.arccos(cos_values)
    weights = actin_flow_result['speed']*rho_flow_result['speed']

    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.hist(theta_values.flatten()/np.pi, bins = 50)
    plt.xlabel(r'|$\mathrm{\theta}$|')
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
    plt.ylabel('Number of pixels')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','angle_value_histograms.pdf'),dpi=300) 

    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.title('Weighted angles')
    plt.hist(theta_values.flatten()/np.pi, bins = 50, weights = weights.flatten(), density = True)
    plt.xlabel(r'|$\mathrm{\theta}$|')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
    plt.ylabel('Density')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','weighted_angle_value_histograms.pdf'),dpi=300) 

    # both_speeds = np.vstack([actin_flow_result['speed'].flatten(),rho_flow_result['speed'].flatten()])
    # color_value = gaussian_kde(both_speeds)(both_speeds)
    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.hist2d(actin_flow_result['speed'][rho_flow_result['speed']>0.01].flatten(), rho_flow_result['speed'][rho_flow_result['speed']>0.01].flatten(), bins = (50,50))
    # plt.scatter(actin_flow_result['speed'].flatten(), rho_flow_result['speed'].flatten(), s= 0.1, c= color_value)
    plt.xlabel('Actin speed [$\mathrm{\mu m}$/s]')
    plt.ylabel('Rho speed [$\mathrm{\mu m}$/s]')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','speed_correlation.png'),dpi=300) 
   
   
#########################################################
# Old code and things I tried but did not use in the end
def make_coexpression_movie(normalised = False):
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    if normalised:
        rho_movie = rho_movie/np.max(rho_movie)*255
        actin_movie = actin_movie/np.max(rho_movie)*255

    joint_movie = np.zeros((rho_movie.shape[0], rho_movie.shape[1], rho_movie.shape[2], 3),dtype = 'int')
    joint_movie[:,:,:,0] = np.round(rho_movie[:,:,:])
    joint_movie[:,:,:,1] = np.round(actin_movie[:,:,:])

    fig = plt.figure(figsize = (2.5,2.5))
    def animate(i): 
        plt.cla()
        plt.imshow(joint_movie[i,:,:], interpolation = None)
        plt.gca().set_axis_off()
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0])
    if normalised:
        filename = 'coexpression_normalised.mp4'
    else:
        filename = 'coexpression_unnormalised.mp4'
    ani.save(os.path.join(os.path.dirname(__file__),'output',filename),dpi=300) 
 
def make_actin_speed_histograms():
    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','fake_flow_result.npy'),allow_pickle='TRUE').item()
    
    plt.figure()
    # plt.hist(actin_flow_result['speed'].flatten()*0.0913/10, bins=100, density=False)
    plt.hist(actin_flow_result['speed'].flatten(), bins=100, density=False)
    plt.xlabel('Actin Speed [$\mathrm{/mu}$/s]')
    plt.ylabel('Number of Pixels')
    plt.tight_layout() 
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','actin_global_speed_histogram.pdf')) 

    print('making histogram for frame')
    print(0)
    first_frame = actin_flow_result['speed'][0,:,:]
    # hist_values, bin_edges = np.histogram(first_frame.flatten()*0.0913/10, bins=50)
    hist_values, bin_edges = np.histogram(first_frame.flatten(), bins=50)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    print('bin centres are')
    print(bin_centres)
    full_data_frame = pd.DataFrame({'Bin center': bin_centres, 'Histogram frame 0': hist_values})
    plt.figure()
    plt.hist(first_frame.flatten()*0.0913/10, bins=50, density=False)
    plt.xlabel('Actin Speed Values [$\mathrm{/mu}$/s]')
    plt.ylabel('Number of Pixels')
    plt.title('Actin speed frame ' + str(0))
    # plt.ylim(0,8200)
    # plt.xlim(0,0.008)
    plt.tight_layout() 
    plt.savefig('actin_speed_histogram_frame_00.png')
 
    for frame_index, frame in enumerate(actin_flow_result['speed'][1:]):
        print('making histogram for frame')
        print(frame_index + 1)
        hist_values, bin_edges = np.histogram(frame.flatten(), bins=bin_edges)
        this_data_frame = pd.DataFrame({'Histogram frame '+ str(frame_index+1): hist_values})
        full_data_frame = pd.concat([full_data_frame,this_data_frame], axis = 1)
        print('bin edges are')
        print(bin_edges)
        plt.figure()
        plt.hist(frame.flatten(), bins=50, density=False)
        plt.xlabel('Actin Speed [$\mathrm{/mu}$/s]')
        plt.ylabel('Number of Pixels')
        plt.title('Actin speed frame ' + str(frame_index + 1))
        # plt.ylim(0,8200)
        # plt.xlim(0,0.008)
        plt.tight_layout() 
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','actin_speed_histogram_frame_' + f"{(frame_index+1):02}" + '.png'))
    
    full_data_frame.to_excel(os.path.join(os.path.dirname(__file__),'output','speed_histograms.xlsx'))
    
    # current_dir = os.path.dirname(__file__)
    # output_location = 
    # compile_command = "ffmpeg -pattern_type glob -i \"output/actin_*.png\" -c:v libx264 -r 30 -pix_fmt yuv420p output/histograms.mp4"
    # subprocess.call(compile_command)

def make_boxsize_comparison():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    # actin_movie = actin_movie[0:3,:,:]
    x_dim = actin_movie.shape[1]
    print(x_dim)
    boxsizes = np.linspace(3, 101, 50)
    # boxsizes = np.linspace(5, 7, 2)
    mean_velocities = np.zeros_like(boxsizes)
    velocities_std = np.zeros_like(boxsizes)
    histogram_figure = plt.figure(figsize = (2.5,2.5))
    def animate(index):
        integer_boxsize = round(boxsizes[index])
        print('calculating boxsize ' + str(integer_boxsize))
        this_result = optical_flow.conduct_optical_flow(actin_movie, boxsize = integer_boxsize, delta_x = 0.0913, delta_t = 10.0)
        this_mean_speed = np.mean(this_result['speed'])
        this_speed_std = np.std(this_result['speed'])
        mean_velocities[index] = this_mean_speed
        velocities_std[index] = this_speed_std
        plt.cla()
        plt.hist(this_result['speed'].flatten(), bins = 50, density = False)
        plt.xlabel('Actin Speed [$\mathrm{\mu m}$/s]')
        plt.ylabel('Number of Pixels')
        plt.xlim(0.0,0.02)
        plt.ylim(0.0,700000)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.title('Boxsize ' + str(integer_boxsize))
        plt.tight_layout()
    animation = FuncAnimation(histogram_figure, animate, frames=len(boxsizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','boxsize_velocity_histograms.mp4'), dpi = 600) 
    

    plt.figure(figsize = (4.5,2.5))
    plt.subplot(121)
    plt.plot(boxsizes.astype('int'), mean_velocities)
    plt.xlabel('boxsize')
    plt.ylabel('mean inferred speed')
    plt.subplot(122)
    plt.plot(boxsizes.astype('int'), velocities_std)
    plt.xlabel('boxsize')
    plt.ylabel('speed standard dev.')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','boxsize_velocities.pdf')) 

def try_dense_optical_flow_on_frame():
    movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    first_frame = movie[3,:,:]
    second_frame = movie[4,:,:]
    
    flow = np.zeros_like(first_frame, dtype = np.uint32)
    
    velocities = cv2.calcOpticalFlowFarneback(first_frame, second_frame, None, 0.5, levels = 1, winsize = 20, iterations = 60, poly_n = 3, poly_sigma = 1.1, flags = 0)
    
    flow_result = dict()
    flow_result['v_x'] = np.array([velocities[:,:,0]])
    flow_result['v_y'] = np.array([velocities[:,:,1]])
    flow_result['original_data'] = movie[3:5,:,:]
    flow_result['delta_x'] = delta_x
    
    x_positions, y_positions, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(flow_result, arrow_boxsize = 15)

    plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    optical_flow.costum_imshow(flow_result['original_data'][1,:,:], delta_x = delta_x)
    plt.quiver(y_positions, x_positions, v_y[0,:,:], -v_x[0,:,:], 
               color = 'magenta',headwidth=5, scale = None)
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','optical_flow_opencv_first_try.pdf'))
    
def conduct_dense_optical_flow():
    movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    # movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    # reader = imageio.get_reader(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control0000.png - Frame Interpolation.mp4'))

    # Determine the dimensions of the video frames
    # num_frames = len(reader)
    # first_frame = reader.get_data(0)
    # height, width,_ = first_frame.shape
    # print(num_frames)
    # print(height)
    # print(width)
    
    # Initialize a 3D NumPy array to store the video frames
    # movie = np.empty((int(num_frames//2e15), height, width), dtype=np.uint8)
    
    # for i, frame in enumerate(reader):
        # if i//2e16==0:
            # movie[i] = frame[:,:,0]
    
    # reader.close()  # Close the video file when done
        # clahed_movie = optical_flow.apply_clahe(movie, clipLimit = 100000)
    clahed_movie = movie[:10]
    
    flow_result = optical_flow.conduct_opencv_flow(clahed_movie, delta_x = delta_x, delta_t = delta_t, smoothing_sigma = 1.3)

    # np.save(os.path.join(os.path.dirname(__file__),'output','opencv_optical_flow_result.npy'), flow_result)
    
    # flow_result['original_data'] = movie
    
    optical_flow.make_velocity_overlay_movie(flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','actin_velocities_opencv.mp4'),arrow_scale = 1.0, arrow_boxsize = 15
                                             )
                                            #  v_max = np.max(flow_result['original_data']))

if __name__ == '__main__':

    ## All figures follow here
    # make_joint_movie()
    # investigate_intensities()
    # make_blurring_analysis(channel = 'actin')
    # make_blurring_analysis(channel = 'rho')
    # investigate_intensity_thresholds()
    # make_thresholded_movies( threshold = 35, rho_sigma = 10, actin_sigma = 10)
    # make_thresholded_movies( threshold = 40, rho_sigma = 10, actin_sigma = 10)
    # make_thresholded_movies( threshold = 30, rho_sigma = 3.0, actin_sigma = 3)
    # make_thresholded_movies( threshold = 30, rho_sigma = 10, actin_sigma = 10)
    # make_thresholded_movies( threshold = 130, rho_sigma = 3, actin_sigma = 3, clahe = 10000)
    # make_thresholded_movies( threshold = 120, rho_sigma = 3, actin_sigma = 3, clahe = 10000)
    # make_thresholded_movies( threshold = 110, rho_sigma = 3, actin_sigma = 3, clahe = 10000)
    # make_thresholded_movies( threshold = 30, rho_sigma = 6, actin_sigma = 6, clahe = None, adaptive = True)
    make_thresholded_movies( threshold = 30, rho_sigma = 3, actin_sigma = 3, clahe = None, adaptive = True)
    # check_error_of_method()
    # check_error_of_method(include_noise = True)
    # make_boxsize_analysis(channel = 'actin')
    # make_OF_blur_analysis(channel = 'actin', boxsize = 21)
    # make_boxsize_analysis(channel = 'actin', blursize = 2.5)
    # make_boxsize_analysis(channel = 'actin', blursize = 3)
    # make_boxsize_analysis(channel = 'actin', blursize = 4)
    # make_boxsize_analysis(channel = 'actin', blursize = 8)
    # make_OF_blur_analysis(channel = 'actin', boxsize = 31)
    # make_OF_blur_analysis(channel = 'rho', boxsize = 31)
    # make_OF_blur_analysis(channel = 'rho', boxsize = 21)
    # make_boxsize_analysis(channel = 'rho', blursize = 2.5)
    # make_boxsize_analysis(channel = 'rho', blursize = 3.0)
    # make_and_save_rho_optical_flow()
    # make_and_save_actin_optical_flow()
    # joint_actin_and_rho_flow_result()
    # joint_actin_and_rho_flow_result(show_blurred=True)
    # visualise_actin_and_rho_velocities()
    # make_joint_speed_and_angle_histograms()

    ## Additional function calls not included in the presentation
    # make_joint_movie(smoothing_sigma = 19)
    # make_actin_speed_histograms()
    # make_boxsize_comparison()
    # make_joint_movie(smoothing_sigma = 19)
    # make_coexpression_movie()
    # make_coexpression_movie(normalised = True)
    # try_dense_optical_flow_on_frame()
    # make_joint_movie(use_clahe = True)
    # make_joint_movie(use_clahe = True)
    # make_joint_movie(smoothing_sigma= 3.0, use_clahe = True)
    # make_actin_clahe_movie()
    # conduct_dense_optical_flow()
    # make_and_save_actin_optical_flow(include_remodelling=True)
    # make_and_save_rho_optical_flow(include_remodelling=True)
    # joint_actin_and_rho_flow_result(include_remodelling=True)
    

