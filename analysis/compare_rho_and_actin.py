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

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

delta_x = 0.0913
delta_t = 10.0

def make_joint_movie(smoothing_sigma = None):
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))

    if smoothing_sigma is not None:
        actin_movie_to_analyse = np.zeros_like(actin_movie, dtype ='double')
        rho_movie_to_analyse = np.zeros_like(rho_movie, dtype ='double')
        for index in range(actin_movie.shape[0]):
            this_actin_frame = actin_movie[index,:,:]
            this_blurred_actin_image = skimage.filters.gaussian(this_actin_frame, sigma =smoothing_sigma, preserve_range = True)
            actin_movie_to_analyse[index,:,:] = this_blurred_actin_image
            this_rho_frame = actin_movie[index,:,:]
            this_blurred_rho_image = skimage.filters.gaussian(this_rho_frame, sigma =smoothing_sigma, preserve_range = True)
            rho_movie_to_analyse[index,:,:] = this_blurred_rho_image
        rho_movie = rho_movie_to_analyse
        actin_movie = actin_movie_to_analyse
 
    fig = plt.figure(figsize = (4.5,2.5))
    def animate(i): 
        # plt.cla()
        plt.subplot(121)
        # plt.gca().set_axis_off()
        plt.title('Rho')
        optical_flow.costum_imshow(rho_movie[i,:,:],delta_x = delta_x)
        plt.subplot(122)
        plt.title('Actin')
        optical_flow.costum_imshow(actin_movie[i,:,:],delta_x = delta_x)
        # plt.imshow(actin_movie[i,:,:],cmap = 'gray_r',vmin = 0, vmax = 255, interpolation = None)
        # plt.gca().set_axis_off()
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
        # plt.savefig(os.path.join(os.path.dirname(__file__),'output','joint_movie' + str(i) + '.png'),dpi=300) 
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','joint_movie_sigma_' + "{:.2f}".format(smoothing_sigma) + '.mp4'),dpi=300) 
    
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
 
def make_blurring_analysis(channel = 'actin'):

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    this_frame = movie[3,:,:]
    this_frame_rgb = np.zeros((this_frame.shape[0], this_frame.shape[1], 3),dtype = 'int')
    this_frame_rgb[:,:,0] = 255 - this_frame
    this_frame_rgb[:,:,1] = 255 - this_frame
    this_frame_rgb[:,:,2] = 255 - this_frame
    
    Xpixels=this_frame.shape[0]#Number of X pixels=379
    Ypixels=this_frame.shape[1]#Number of Y pixels=279
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
        # ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan =2)
        ax1 = fig.add_subplot(outer_grid[0,0], anchor = (0.75,1.0))
        ax1.imshow(this_frame_rgb, extent = [0,y_extent, x_extent, 0], interpolation = None )
        ax1.set_xlabel("y-position [$\mathrm{\mu}$m]")
        ax1.set_ylabel("x-position [$\mathrm{\mu}$m]")
        # ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan =2)
        ax2 = fig.add_subplot(outer_grid[0,1])
        # plt.imshow(this_filtered_image, cmap = 'gray_r', extent = [0,y_extent, x_extent, 0], interpolation = None )
        # plt.xlabel("y-position [$\mathrm{\mu}$m]")
        # plt.ylabel("x-position [$\mathrm{\mu}$m]")
        optical_flow.costum_imshow(this_filtered_image, delta_x = delta_x)
        # ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=1, rowspan =1)
        ax3 = fig.add_subplot(left_grid[0,0])
        # ax3.plot(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.5, linewidth = 0.1)
        # ax3.plot(y_positions_on_line,this_filtered_line, linewidth =0.1)
        ax3.step(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.6, linewidth = 0.15)
        ax3.step(y_positions_on_line,this_filtered_line, linewidth =0.2, where = 'mid')
        ax3.set_ylabel('Intensity')
        # ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=1, rowspan =1)
        ax4 = fig.add_subplot(left_grid[1,0])
        # ax4.plot(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.5, linewidth = 0.1)
        # ax4.plot(y_positions_on_line,this_filtered_line, linewidth =0.1)
        ax4.step(y_positions_on_line,highlighted_line, color = 'green', alpha = 0.6, linewidth = 0.15, where = 'mid' )
        ax4.step(y_positions_on_line,this_filtered_line, linewidth =0.2, where = 'mid')
        ax4.set_xlabel('y-position [$\mathrm{\mu}$m]')
        ax4.set_ylabel('Intensity')
        ax4.set_xlim(5,15)
        ax5 = fig.add_subplot(outer_grid[1,1], anchor = (0.5,4.5), box_aspect = 1.1)
        # ax4 = plt.subplot2grid((4, 2), (2, 1), colspan=1, rowspan =2)
        ax5.bar(bin_centers, this_histogram, width = 1.0)
        ax5.bar(bin_centers, original_histogram, width = 1.0, color = 'grey', alpha = 0.8)
        # ax4.hist(this_frame.flatten(),bins=255, range = (0,255), color = 'grey', alpha = 0.3, density = True)
        # ax5.hist(this_filtered_image.flatten(),bins=255, range = (0,255), density = True )
        ax5.set_xlim(0,120)
        ax5.set_xlabel('Image intensity')
        ax5.set_ylabel('Normalised occupancy')
        # ax5_bounds = ax5.get_position().bounds
        # ax5.set_position([ax5_bounds[0], ax5_bounds[1], ax5_bounds[2], ax5_bounds[3]])
        # if index <1:
            # plt.tight_layout()
    animation = FuncAnimation(fig, animate_blurr, frames=len(blur_sizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','blur_analysis_' + channel + '.mp4'),dpi=300) 
    
def make_and_save_rho_optical_flow():
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(rho_movie, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = 2.5)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'), this_result)

def make_and_save_actin_optical_flow():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(actin_movie, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = 2.5)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'), this_result)
    
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

# ffmpeg -pattern_type glob -i "actin_*.png" -c:v libx264 -r 30 -pix_fmt yuv420p histograms.mp4
def visualise_actin_and_rho_velocities():
    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(actin_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','actin_velocities.mp4'),arrow_scale = 1.0, arrow_boxsize = 15)

    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(rho_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','rho_velocities.mp4'), arrow_boxsize = 15)

@jit(nopython = True)
def make_fake_data_frame(x_position, y_position):
    x_dim = 1000
    x = np.linspace(0,20,x_dim)#(0,10,100)0-10 100 samples uniform,np.linespace(start,stop,number)
    y = np.linspace(0,20,x_dim)
    frame = np.zeros((x_dim,x_dim))
    for x_index, x_value in enumerate(x):
        for y_index,y_value in enumerate(y):
            frame[x_index, y_index] = np.exp(-(x_value-x_position)**2 - (y_value-y_position)**2)
    delta_x = x[1]-x[0]
    frame += np.random.rand(x_dim, x_dim)*0.0000001
    frame = np.abs(frame)

    return frame, delta_x

def check_error_of_method():
    
    x_velocity = 0.1
    y_velocity = 0.2
    delta_t = 0.5

    x_step = x_velocity*delta_t
    y_step = y_velocity*delta_t  
    n_steps = 5
    frames = []
    x_position = 5
    y_position = 3
    for frame_index in range(n_steps):
        this_frame, delta_x = make_fake_data_frame(x_position, y_position)
        frames.append(this_frame)
        x_position += x_step
        y_position += y_step
        
    fake_data = np.array(frames)

    this_result = optical_flow.conduct_optical_flow(fake_data, boxsize = 15, delta_x = delta_x, delta_t = 0.5)
    np.save(os.path.join(os.path.dirname(__file__),'output','fake_flow_result.npy'), this_result)


    
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
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','fake_v_histogram.pdf')) 
    


    # this_result['original_data'][:-1][this_result['v_x'] == np.inf] = 1.0
    # this_result['original_data'][:-1][this_result['v_x'] != np.inf] = 0.0
    optical_flow.make_velocity_overlay_movie(this_result, 
                                             os.path.join(os.path.dirname(__file__),'output','made_up_data_velocities.mp4'),
                                             boxsize = 40,
                                             arrow_scale = 0.2)

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

def investigate_intensities():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    
    plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    plt.subplot(121)
    plt.hist(actin_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Actin intensity value')
    plt.ylabel('Number of pixels')
    
    plt.subplot(122)
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    plt.hist(rho_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Rho intensity value')
    plt.ylabel('Number of pixels')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','both_intensity_histgrams.pdf'))
 
def make_OF_blur_analysis(channel = 'actin'):

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    # blur_sizes = np.arange(0.1,5,0.01)
    # blur_sizes = np.arange(0.1,5.0,0.01)
    blur_sizes = np.arange(0.5,15,0.1)
    
    mean_velocities = np.zeros_like(blur_sizes, dtype = 'float')
    velocities_std = np.zeros_like(blur_sizes, dtype = 'float')

    fig = plt.figure(figsize = (4.5,4.5), constrained_layout = True)
    def animate_blursizes(index):
        plt.clf()
        blursize = blur_sizes[index]
        blursize_in_micro_meters = blursize*delta_x
        fig.suptitle('Blur $\mathrm{\sigma}$=' + "{:.2f}".format(blursize) + r' ($\approx$' + "{:.2f}".format(blursize_in_micro_meters) + '$\mathrm{\mu}$m)' )
        this_result = optical_flow.conduct_optical_flow(movie[3:5,:,:], boxsize = 21, delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = blursize)
        cos_values = this_result['v_y']/this_result['speed']
        angles = np.arccos(cos_values)*np.sign(this_result['v_x'])
        x_positions, y_positions, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(this_result, arrow_boxsize = 15)
        this_mean_speed = np.mean(this_result['speed'])
        this_speed_std = np.std(this_result['speed'])
        mean_velocities[index] = this_mean_speed
        velocities_std[index] = this_speed_std

        outer_grid = fig.add_gridspec(2,2)
        right_grid = outer_grid[1,1].subgridspec(2,1)

        ax1 = fig.add_subplot(outer_grid[0,0])
        ax1.set_title('blur hidden')
        optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
        ax1.quiver(y_positions, x_positions, v_x[0,:,:], -v_y[0,:,:], color = 'magenta',headwidth=5, scale = None)

        ax2 = fig.add_subplot(outer_grid[0,1])
        ax2.set_title('blur visible')
        optical_flow.costum_imshow(this_result['blurred_data'][1,:,:], delta_x = delta_x)
        ax2.quiver(y_positions, x_positions, v_x[0,:,:], -v_y[0,:,:], color = 'magenta',headwidth=5, scale = None)

        ax2 = fig.add_subplot(outer_grid[1,0])
        ax2.hist(this_result['speed'].flatten(), bins = 50, density = False)
        ax2.ticklabel_format(scilimits = (-3,3))
        ax2.set_xlabel('Actin Speed [$\mathrm{\mu m}$/s]')
        # plt.xlim((0,0.02))
        ax2.set_ylabel('Number of Pixels')
        
        ax3 = fig.add_subplot(right_grid[0,0])
        ax3.hist(angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
        ax3.ticklabel_format(scilimits = (-3,3))
        # ax3.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        # ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        ax3.set_xlabel('Angle to y axis')
        ax3.set_ylabel('# Pixels')
        
        ax4 = fig.add_subplot(right_grid[1,0], sharex = ax3)
        ax4.hist(angles.flatten()/np.pi, bins = 50, density = False, weights = this_result['speed'].flatten())
        ax4.ticklabel_format(scilimits = (-3,3))
        ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        ax4.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        ax4.set_xlabel('Weighted angle to y axis')
        ax4.set_ylabel('# Pixels')
        
    animation = FuncAnimation(fig, animate_blursizes, frames=len(blur_sizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','blursize_analysis_' + channel + '.mp4'),dpi=300) 

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
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','blursize_velocities.pdf')) 
 
def make_boxsize_analysis(channel = 'actin'):

    if channel == 'rho':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    elif channel == 'actin':
        movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    boxsizes = np.arange(5,150,2).astype('int')
    # boxsizes = np.arange(5,56,50).astype('int')
    
    mean_velocities = np.zeros_like(boxsizes, dtype = 'float')
    velocities_std = np.zeros_like(boxsizes, dtype = 'float')

    fig = plt.figure(figsize = (4.5,4.5), constrained_layout = True)
    def animate_boxsizes(index):
        plt.clf()
        boxsize = boxsizes[index]
        boxsize_in_micro_meters = boxsize*delta_x
        fig.suptitle('Boxsize b=' + str(boxsize) + r' ($\approx$' + "{:.2f}".format(boxsize_in_micro_meters) + '$\mathrm{\mu}$m)' )
        this_result = optical_flow.conduct_optical_flow(movie[3:5,:,:], boxsize = boxsizes[index], delta_x = 0.0913, delta_t = 10.0, smoothing_sigma = 1.3)
        cos_values = this_result['v_y']/this_result['speed']
        angles = np.arccos(cos_values)*np.sign(this_result['v_x'])
        x_positions, y_positions, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(this_result, arrow_boxsize = 15)
        this_mean_speed = np.mean(this_result['speed'])
        this_speed_std = np.std(this_result['speed'])
        mean_velocities[index] = this_mean_speed
        velocities_std[index] = this_speed_std

        plt.subplot(221)
        optical_flow.costum_imshow(movie[4,:,:], delta_x = delta_x)
        plt.quiver(y_positions, x_positions, v_x[0,:,:], -v_y[0,:,:], color = 'magenta',headwidth=5, scale = None)

        plt.subplot(222)
        plt.hist(this_result['speed'].flatten(), bins = 50, density = False)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.xlabel('Actin Speed [$\mathrm{\mu m}$/s]')
        # plt.xlim((0,0.02))
        plt.ylabel('Number of Pixels')
        
        plt.subplot(223)
        plt.hist(angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        plt.xlabel('Angle to y axis')
        plt.ylabel('Number of Pixels')
        
        plt.subplot(224)
        plt.hist(angles.flatten()/np.pi, bins = 50, density = False, weights = this_result['speed'].flatten())
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
        plt.xlabel('Weighted angle to y axis')
        plt.ylabel('Number of Pixels')
        
    animation = FuncAnimation(fig, animate_boxsizes, frames=len(boxsizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','boxsize_analysis_' + channel + '.mp4'),dpi=300) 

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
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','new_boxsize_velocities.pdf')) 
 
if __name__ == '__main__':
    # make_and_save_rho_optical_flow()
    # make_and_save_actin_optical_flow()
    # visualise_actin_and_rho_velocities()
    make_boxsize_analysis()
    make_OF_blur_analysis()
    # investigate_actin_intensity()

    # make_actin_speed_histograms()
    # check_error_of_method()
    # make_boxsize_comparison()
    # make_joint_movie(smoothing_sigma = 19)
    # make_coexpression_movie()
    # make_coexpression_movie(normalised = True)
    # make_blurring_analysis(channel = 'actin')
    # make_blurring_analysis(channel = 'rho')
