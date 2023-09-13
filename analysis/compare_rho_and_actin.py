import sys
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import tifffile
from numba import jit

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def make_and_save_rho_optical_flow():
    
    rho_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(rho_movie)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'), this_result)

def make_and_save_actin_optical_flow():
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    this_result = optical_flow.conduct_optical_flow(actin_movie)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'), this_result)
    
def make_actin_speed_histograms():
    actin_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','actin_optical_flow_result.npy'),allow_pickle='TRUE').item()
    
    plt.figure()
    plt.hist(actin_flow_result['speed'].flatten()*0.0913/10, bins=100, density=False)
    plt.xlabel('Actin Speed Values')
    plt.ylabel('Number of Pixels')
    plt.tight_layout() 
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','actin_global_speed_histogram.pdf')) 

    print('making histogram for frame')
    print(0)
    first_frame = actin_flow_result['speed'][0,:,:]
    hist_values, bin_edges = np.histogram(first_frame.flatten()*0.0913/10, bins=50)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    print('bin centres are')
    print(bin_centres)
    full_data_frame = pd.DataFrame({'Bin center': bin_centres, 'Histogram frame 0': hist_values})
    plt.figure()
    plt.hist(first_frame.flatten()*0.0913/10, bins=50, density=False)
    plt.xlabel('Actin Speed Values')
    plt.ylabel('Number of Pixels')
    plt.title('Actin speed frame ' + str(0))
    plt.ylim(0,8200)
    plt.xlim(0,0.008)
    plt.tight_layout() 
    plt.savefig('actin_speed_histogram_frame_00.png')
 
    for frame_index, frame in enumerate(actin_flow_result['speed'][1:]):
        print('making histogram for frame')
        print(frame_index + 1)
        hist_values, bin_edges = np.histogram(frame.flatten()*0.0913/10, bins=bin_edges)
        this_data_frame = pd.DataFrame({'Histogram frame '+ str(frame_index+1): hist_values})
        full_data_frame = pd.concat([full_data_frame,this_data_frame], axis = 1)
        print('bin edges are')
        print(bin_edges)
        plt.figure()
        plt.hist(frame.flatten()*0.0913/10, bins=50, density=False)
        plt.xlabel('Actin Speed Values')
        plt.ylabel('Number of Pixels')
        plt.title('Actin speed frame ' + str(frame_index + 1))
        plt.ylim(0,8200)
        plt.xlim(0,0.008)
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
                                             os.path.join(os.path.dirname(__file__),'output','actin_velocities.mp4'))

    rho_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','rho_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.make_velocity_overlay_movie(rho_flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','rho_velocities.mp4'))

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
    frame += np.random.rand(x_dim, x_dim)*0.00001
    frame = np.abs(frame)

    return frame, delta_x

def check_error_of_method():
    
    x_step = 0.1
    y_step = 0.2  
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

    this_result = optical_flow.conduct_optical_flow(fake_data, boxsize = 15, delta_x = delta_x)
    
    # this_result['v_x'][this_result['v_x'] == np.inf] = 0.0
    print('max vx is')
    print(np.nanmax(this_result['v_x']))
    print('min vx is')
    print(np.nanmin(this_result['v_x']))
    print('mean vx is')
    print(np.nanmean(this_result['v_x']))

    # this_result['v_y'][this_result['v_y'] == np.inf] = 0.0
    print('max vy is')
    print(np.nanmax(this_result['v_y']))
    print('min vy is')
    print(np.nanmin(this_result['v_y']))
    print('mean vy is')
    print(np.nanmean(this_result['v_y']))
    
    print('true vx is 0.1')
    print('true vy is 0.2')
 
    # this_result['original_data'][:-1][this_result['v_x'] == np.inf] = 1.0
    # this_result['original_data'][:-1][this_result['v_x'] != np.inf] = 0.0
    optical_flow.make_velocity_overlay_movie(this_result, 
                                             os.path.join(os.path.dirname(__file__),'output','made_up_data_velocities.mp4'),
                                             boxsize = 40,
                                             arrow_scale = 0.2)

if __name__ == '__main__':
    # make_and_save_rho_optical_flow()
    # make_and_save_actin_optical_flow()
    # visualise_actin_and_rho_velocities()
    #make_actin_speed_histograms()
    check_error_of_method()
