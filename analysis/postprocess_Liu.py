import scipy.io
import os
import compare_rho_and_actin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import skimage
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow


delta_x = compare_rho_and_actin.delta_x
delta_t = compare_rho_and_actin.delta_t

def convert_Liu_result(Liu_result):
    v_x = Liu_result['u_x']*delta_x/delta_t
    v_y = Liu_result['v_original']*delta_x/delta_t
    
    new_x_locations = np.zeros((len(x_locations), x_locations[0][0].shape[0],x_locations[0][0].shape[1] ))
    new_y_locations = np.zeros_like(new_x_locations)
    new_v_x = np.zeros_like(new_x_locations)
    new_v_y = np.zeros_like(new_x_locations)

    for frame_index in range(len(x_locations)):
        new_x_locations[frame_index,:,:] = x_locations[frame_index][0]
        new_y_locations[frame_index,:,:] = y_locations[frame_index][0]
        new_v_x[frame_index,:,:] = v_x[frame_index][0]
        new_v_y[frame_index,:,:] = v_y[frame_index][0]

    return new_x_locations, new_y_locations, new_v_x, new_v_y

def visualise_Liu_result():
    Liu_result = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'data', 'Liu_method.mat'))
    # x_locations, y_locations, v_x, v_y = convert_Liu_result(Liu_result)
    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    v_x = np.zeros(1,actin_movie.shape[1], actin_movie.shape[2]) 
    v_y = np.zeros(1,actin_movie.shape[1], actin_movie.shape[2]) 
    v_x[0,:,:] = Liu_result['ux']
    v_y[0,:,:] = Liu_result['uy']
    speed = np.sqrt(v_x**2 + v_y**2)
    
    flow_result = dict()
    flow_result['v_x'] = v_x
    flow_result['v_y'] = v_y
    flow_result['delta_x'] = delta_x
    flow_result['delta_t'] = delta_t

    x_locations, y_loctations, v_x, v_y = optical_flow.subsample_velocities_for_visualisation(flow_result)

    # actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    actin_movie_blurred = optical_flow.blur_movie(actin_movie, smoothing_sigma = 3)

    fig = plt.figure(figsize = (2.5,2.5))
    optical_flow.costum_imshow(actin_movie[i,:,:], delta_x = delta_x, autoscale=False)
    plt.quiver( x_locations[i,:,:],y_locations[i,:,:], v_x[i,:,:],-v_y[i,:,:], color = 'magenta',headwidth=5, scale = 1.0)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=x_locations.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')

    saving_name = os.path.join(os.path.dirname(__file__),'output', 'Liu_visualisation.mp4')
    ani.save(saving_name,dpi=600) 
    
    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.hist(speed.flatten(), bins = 50)
    plt.xlabel('speed')
    plt.ylabel('number of boxes')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output', 'Liu_speed_histogram.pdf'))

if __name__ == '__main__':
    visualise_Liu_result()