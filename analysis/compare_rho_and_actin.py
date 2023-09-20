import sys
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
import pandas as pd
from numba import jit
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def make_and_save_rho_optical_flow():
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(rho_movie, delta_x = 0.0913, delta_t = 10.0)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'), this_result)

def make_and_save_actin_optical_flow():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(actin_movie, delta_x = 0.0913, delta_t = 10.0)
    
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
                                             os.path.join(os.path.dirname(__file__),'output','actin_velocities.mp4'),arrow_scale = 5.0, boxsize = 15)

    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(rho_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','rho_velocities.mp4'), boxsize = 15, arrow_scale = 5.0)

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
    actin_movie = actin_movie[0:3,:,:]
    x_dim = actin_movie.shape[1]
    # boxsizes = np.linspace(5, x_dim, 3)
    boxsizes = np.linspace(5, 7, 2)
    mean_velocities = np.zeros_like(boxsizes)
    velocities_std = np.zeros_like(boxsizes)
    histogram_figure = plt.figure(figsize = (2.5,2.5))
    def animate(index):
        integer_boxsize = int(boxsizes[index])
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
        plt.xlim(0.0,0.05)
        plt.ylim(0.0,20000)
        plt.gca().ticklabel_format(scilimits = (-3,3))
        plt.title('Boxsize ' + str(integer_boxsize))
        plt.tight_layout()
    animation = FuncAnimation(histogram_figure, animate, frames=len(boxsizes))
    animation.save(os.path.join(os.path.dirname(__file__),'output','boxsize_velocity_histograms.mp4'), dpi = 600) 
    
    print(velocities_std)

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

def investigate_actin_intensity():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    plt.figure(figsize = (2.5,2.5))
    plt.hist(actin_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Actin intensity value')
    plt.ylabel('number of pixels')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','actin_intensity_histogram.pdf'))
    
    mean_intensity = np.mean(actin_movie)
    print('mean intensity is')
    print(mean_intensity)
    print('stdev of intensity is')
    print(np.std(actin_movie))
    mean_intensity = np.mean(actin_movie)
    
    unique_intensity_values = np.unique(actin_movie)
    print('there are ' + str(len(unique_intensity_values)) + ' unique intensity values present in the image')
    print('these are')
    print(unique_intensity_values)

    cmap = 'gray_r'
    v_min = 0
    v_max = 255
    Xpixels=actin_movie.shape[1]#Number of X pixels=379
    Ypixels=actin_movie.shape[2]#Number of Y pixels=279
    x_extent = Xpixels * 0.0913
    y_extent = Ypixels * 0.0913
    plt.figure(figsize = (2.5,2.5))
    plt.imshow(actin_movie[0,:,:],cmap = cmap, extent = [0,y_extent, x_extent, 0], vmin = v_min, vmax = v_max, interpolation = None)
    plt.xlabel("y-position [$\mathrm{\mu}$m]")
    plt.ylabel("x-position [$\mathrm{\mu}$m]")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','actin_frame_0.png'), dpi = 600)

    # also make a histogram for rho
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    plt.figure(figsize = (2.5,2.5))
    plt.hist(rho_movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Rho intensity value')
    plt.ylabel('number of pixels')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','rho_intensity_histogram.pdf'))
 
if __name__ == '__main__':
    # make_and_save_rho_optical_flow()
    # make_and_save_actin_optical_flow()
    # visualise_actin_and_rho_velocities()
    # investigate_actin_intensity()
    # make_actin_speed_histograms()
    # check_error_of_method()
    make_boxsize_comparison()
