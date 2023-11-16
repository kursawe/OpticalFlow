import scipy.io
import os
import compare_rho_and_actin
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import skimage
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)

delta_x = compare_rho_and_actin.delta_x
delta_t = compare_rho_and_actin.delta_t

def convert_PIV_result(PIV_result):
    x_locations = PIV_result['x']*delta_x
    y_locations = PIV_result['y']*delta_x
    v_x = PIV_result['u_original']*delta_x/delta_t
    v_y = PIV_result['v_original']*delta_x/delta_t
    
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

def threshold_PIV_result(x_locations, y_locations, v_x, v_y, speed, raw_movie):
    movie_blurred = optical_flow.blur_movie(raw_movie, smoothing_sigma = 3.0)
    # threshold_masks = movie_blurred < 30
    threshold_masks = np.logical_not(optical_flow.apply_adaptive_threshold(movie_blurred, window_size = 151, threshold = -5))

    for frame_index in range(len(x_locations)):
        these_x_indices = np.int64(x_locations[frame_index,:,:]/delta_x).flatten()
        these_y_indices = np.int64(y_locations[frame_index,:,:]/delta_x).flatten()
        for vector_index in range(len(these_x_indices)):
            x_index = these_x_indices[vector_index]
            y_index = these_y_indices[vector_index]
            threshold_mask_here = threshold_masks[frame_index, y_index, x_index]
            if threshold_mask_here:
                v_x[frame_index,:,:][np.unravel_index(vector_index, v_x[frame_index,:,:].shape)] = 0.0
                v_y[frame_index,:,:][np.unravel_index(vector_index, v_y[frame_index,:,:].shape)] = 0.0
                speed[frame_index,:,:][np.unravel_index(vector_index, v_y[frame_index,:,:].shape)] = 0.0
            if (speed[frame_index,:,:][np.unravel_index(vector_index, v_x[frame_index,:,:].shape)] < 0.01 or 
                 speed[frame_index,:,:][np.unravel_index(vector_index, v_x[frame_index,:,:].shape)] > 0.08):
                v_x[frame_index,:,:][np.unravel_index(vector_index, v_x[frame_index,:,:].shape)] = 0.0
                v_y[frame_index,:,:][np.unravel_index(vector_index, v_y[frame_index,:,:].shape)] = 0.0
                speed[frame_index,:,:][np.unravel_index(vector_index, v_y[frame_index,:,:].shape)] = 0.0



def visualise_PIV_result(channel = 'actin'):
    if channel == 'actin':
        loading_path = os.path.join(os.path.dirname(__file__), 'data', 'PIVlab_actin_new.mat')
        original_data_path = os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif')
        suffix = 'actin'
    elif channel == 'rho':
        loading_path = os.path.join(os.path.dirname(__file__), 'data', 'PIVlab_rho.mat')
        original_data_path = os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif')
        suffix = 'rho'
    PIV_result = scipy.io.loadmat(loading_path)
    x_locations, y_locations, v_x, v_y = convert_PIV_result(PIV_result)
    speed = np.sqrt(v_x**2 + v_y**2)
    

    actin_movie = skimage.io.imread(original_data_path)

    threshold_PIV_result(x_locations, y_locations, v_x, v_y, speed, actin_movie)
    fig = plt.figure(figsize = (2.5,2.5))
    def animate(i): 
        plt.cla()
        optical_flow.costum_imshow(actin_movie[i,:,:], delta_x = delta_x, autoscale=False)
        plt.quiver( x_locations[i,:,:],y_locations[i,:,:], v_x[i,:,:],-v_y[i,:,:], color = 'magenta',headwidth=5, scale = 1.0)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=x_locations.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')

    saving_name = os.path.join(os.path.dirname(__file__),'output', 'piv_visualisation_' + suffix + '.mp4')
    ani.save(saving_name,dpi=600) 
    
    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.hist(speed.flatten(), bins = 50)
    plt.xlabel('speed')
    plt.ylabel('number of boxes')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output', 'piv_speed_histogram_' + suffix + '.pdf'))
    
def visualise_joint_PIV_result():
    actin_loading_path = os.path.join(os.path.dirname(__file__), 'data', 'PIVlab_actin_new.mat')
    actin_original_data_path = os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif')
    rho_loading_path = os.path.join(os.path.dirname(__file__), 'data', 'PIVlab_rho.mat')
    rho_original_data_path = os.path.join(os.path.dirname(__file__),'data','Rho1-reporter_MB160918_20_a_control.tif')
    PIV_rho_result = scipy.io.loadmat(rho_loading_path)
    PIV_actin_result = scipy.io.loadmat(actin_loading_path)
    x_locations_rho, y_locations_rho, v_x_rho, v_y_rho = convert_PIV_result(PIV_rho_result)
    x_locations_actin, y_locations_actin, v_x_actin, v_y_actin = convert_PIV_result(PIV_actin_result)
    speed_rho = np.sqrt(v_x_rho**2 + v_y_rho**2)
    speed_actin = np.sqrt(v_x_actin**2 + v_y_actin**2)

    actin_movie = skimage.io.imread(actin_original_data_path)
    rho_movie = skimage.io.imread(rho_original_data_path)

    threshold_PIV_result(x_locations_actin, y_locations_actin, v_x_actin, v_y_actin, speed_actin, actin_movie)
    threshold_PIV_result(x_locations_rho, y_locations_rho, v_x_rho, v_y_rho, speed_rho, rho_movie)

    test_location = np.array([12.5,7])
    
    x_location_index = np.argmin(np.abs(x_locations_actin[0,:,:] - test_location[1]))
    x_location_index_2D = np.unravel_index(x_location_index, x_locations_actin[0,:,:].shape)
    y_location_index = np.argmin(np.abs(y_locations_actin[0,:,:] - test_location[0]))
    y_location_index_2D = np.unravel_index(y_location_index, y_locations_actin[0,:,:].shape)
    
    print('actin speed at test location is')
    print(speed_actin[3,4,7])
    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(121)
        plt.title('Rho')
        optical_flow.costum_imshow(rho_movie[i,:,:],autoscale = False, delta_x = delta_x)
        plt.quiver( x_locations_rho[i,:,:],y_locations_rho[i,:,:], v_x_rho[i,:,:],-v_y_rho[i,:,:], color = 'magenta',headwidth=5, scale = 1.0)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        plt.subplot(122)
        plt.title('Actin')
        optical_flow.costum_imshow(actin_movie[i,:,:],autoscale = False, delta_x = delta_x)
        plt.quiver( x_locations_actin[i,:,:],y_locations_actin[i,:,:], v_x_actin[i,:,:],-v_y_actin[i,:,:], color = 'magenta',headwidth=5, scale = 1.0)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    ani = FuncAnimation(fig, animate, frames=rho_movie.shape[0] - 1)
    ani.save(os.path.join(os.path.dirname(__file__),'output','joint_movie_PIV.mp4'),dpi=300) 

    assert(np.all(x_locations_rho == x_locations_actin))
    assert(np.all(y_locations_rho == y_locations_actin))
    

    rho_cos_values = v_y_rho/speed_rho
    rho_angles = np.arccos(rho_cos_values)*np.sign(v_x_rho)

    actin_cos_values = v_y_actin/speed_actin
    actin_angles = np.arccos(actin_cos_values)*np.sign(v_x_actin)
    
    speed_actin_unfiltered = np.copy(speed_actin)
    speed_rho_unfiltered = np.copy(speed_rho)

    speed_actin = speed_actin[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    speed_rho = speed_rho[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    actin_cos_values = actin_cos_values[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    rho_cos_values = rho_cos_values[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    actin_angles = actin_angles[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    rho_angles = rho_angles[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    
    # import pdb; pdb.set_trace()

    plt.figure(figsize = (4.5,4.5), constrained_layout = True)

    plt.subplot(221)
    plt.hist(speed_actin.flatten(), bins = 50, range = (0,0.1))
    plt.ylabel('Number of boxes')
    plt.xlabel('Speed [$\mathrm{\mu m}$/s]')
    plt.title('Actin')
    plt.gca().ticklabel_format(scilimits = (-3,3))

    plt.subplot(222)
    plt.title('Rho')
    plt.hist(speed_rho.flatten(), bins = 50, range = (0,0.1))
    plt.ylabel('Number of boxes')
    plt.xlabel('Speed [$\mathrm{\mu m}$/s]')
    plt.gca().ticklabel_format(scilimits = (-3,3))

    plt.subplot(223)
    plt.hist(actin_angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    plt.xlabel('Angle to y axis')
    plt.ylabel('Number of boxes')
 
    plt.subplot(224)
    plt.hist(rho_angles.flatten()/np.pi, bins = 50, range = (-1,1), density = False)
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    plt.xlabel('Angle to y axis')
    plt.ylabel('Number of boxes')

    plt.savefig(os.path.join(os.path.dirname(__file__),'output','joint_speed_histograms_PIV.pdf'),dpi=300) 
    
    scalar_products = (v_x_actin*v_x_rho +
                       v_y_actin*v_y_rho)
    
    cos_values = scalar_products/(speed_actin_unfiltered*speed_rho_unfiltered)
    theta_values = np.arccos(cos_values)
    weights = speed_actin_unfiltered*speed_rho_unfiltered
    cos_values = cos_values[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    theta_values = theta_values[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]
    weights = weights[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]


    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.hist(theta_values.flatten()/np.pi, bins = 50)
    plt.xlabel(r'|$\mathrm{\theta}$|')
    plt.gca().ticklabel_format(scilimits = (-3,3))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
    plt.ylabel('Number of boxes')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','angle_value_histograms_PIV.pdf'),dpi=300) 

    plt.figure(figsize = (2.5, 2.5), constrained_layout = True)
    plt.title('Weighted angles')
    plt.hist(theta_values.flatten()/np.pi, bins = 50, weights = weights.flatten(), density = True)
    plt.xlabel(r'|$\mathrm{\theta}$|')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'%g$\mathrm{\pi}$'))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
    plt.ylabel('Density')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','weighted_angle_value_histograms_PIV.pdf'),dpi=300) 

    plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    ax = plt.subplot(projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_thetalim((0.0,np.pi))
    ax.set_xticks(np.linspace(0, np.pi, 5))  # Set tick positions
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°'])  # Set tick labels
    ax.hist(theta_values.flatten(), bins = 20)
    ax.text(-0.05, 0.5, 'Number of Boxes', rotation='vertical', va='center', ha='center', transform=ax.transAxes)
    plt.title('Angle between rho\n and actin motion')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','polar_histogram_PIV.pdf'),dpi=300) 


    # both_speeds = np.vstack([actin_flow_result['speed'].flatten(),rho_flow_result['speed'].flatten()])
    # color_value = gaussian_kde(both_speeds)(both_speeds)
    plt.figure(figsize = (3.5, 2.5), constrained_layout = True)
    plt.hist2d(speed_actin[speed_rho>0.01].flatten(), speed_rho[speed_rho>0.01].flatten(), bins = (50,50))
    plt.xlabel('Actin speed [$\mathrm{\mu m}$/s]')
    plt.ylabel('Rho speed [$\mathrm{\mu m}$/s]')
    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('number of boxes')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','speed_correlation.png'),dpi=300) 
   
 

if __name__ == '__main__':
    # visualise_PIV_result(channel = 'actin')
    # visualise_PIV_result(channel = 'rho')
    visualise_joint_PIV_result()