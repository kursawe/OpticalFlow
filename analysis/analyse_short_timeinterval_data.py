import os
import re
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
import pandas as pd
from matplotlib.animation import FuncAnimation
import skimage
import scipy.io
import tifffile

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow

def read_image_sequence(path_to_images):
    '''
    A function to read in images from a specified folder and store them in an array.
    
    Parameters
    ---------- 
    path_to_images: string
    Path to the folder containing the images, e.g. 'C:/Users/this_user/data/image_folder/'
    
    Output
    ------
    A list of image arrays.
    '''
    
    #List the names of the files in the specified folder numerically.
    list_of_file_paths = list_file_paths_in_folder_numerically(path_to_images)
    
    #Make an empty list to store the images.
    images=[]
    
    #Read in the input files and append them to the list. 
    #cv2.IMREAD_UNCHANGED ensures that the properties of the images (e.g. bit depth and number of channels) are not changed.        
    for frame_index in range(len(list_of_file_paths)):
        image = cv2.imread(list_of_file_paths[frame_index], cv2.IMREAD_UNCHANGED)
        images.append(image)
    
    #Return the array of images.   
    return images



def list_file_paths_in_folder_numerically(path_to_folder):
    '''
    Numerically list file paths in the specified folder.
    
    Parameters
    ---------- 
    path_to_folder: string 
    Path to the folder containing the images, e.g. 'C:/Users/this_user/data/folder/'
    
    Output
    ------
    A 1D numpy array of numerically sorted file paths.
    '''
    
    #Make an empty array to store the file names.
    file_paths= []
        
    #Append the path to each file in the specified folder to the list of file names.
    #Loop through the names of all of the items in the folder.
    for files in os.listdir(path_to_folder):
        #If a given item is a file,
        if os.path.isfile(os.path.join(path_to_folder, files)):
            #Get its path,
            file_path = os.path.join(path_to_folder, files)
            #and append the path to the list.
            file_paths.append(file_path)
                
    #Sort the file paths numerically.
    sorted_file_paths = sort_filenames_numerically(file_paths)
    
    #Return the sorted file paths.
    return sorted_file_paths



def sort_filenames_numerically(list_of_file_names):
    '''
    Sort a list of file names numerically.
    '''      
    numerically_sorted_filenames=sorted(list_of_file_names, key=numerical_sorting_function)       
    return numerically_sorted_filenames

def numerical_sorting_function(filename):
    '''
    A helper function which sorts file names numerically. 
        
    '''
    nondigits = re.compile("\D")
    return int(nondigits.sub("", filename)) 

def conduct_dense_optical_flow_one_frame():
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    
    delta_x = 1.0
    delta_t = 1.0
    measurements = pd.read_excel(os.path.join(os.path.dirname(__file__),'data','displacement_measurements.xlsx'))
    x_coordinates_start = measurements['x-position start'].values
    x_coordinates_end = measurements['x-position end'].values
    y_coordinates_start = measurements['y-position start'].values
    y_coordinates_end = measurements['y-position end'].values
 
    # movie = movie[8:10,:,:]
    
        # clahed_movie = optical_flow.apply_clahe(movie, clipLimit = 100000)
    
    flow_result = optical_flow.conduct_opencv_flow(movie, delta_x = delta_x, delta_t = delta_t, smoothing_sigma = None,
                                                   levels = 5, winsize = 50, iterations = 40, poly_n = 5, poly_sigma = 1.1)

    # np.save(os.path.join(os.path.dirname(__file__),'output','opencv_optical_flow_result.npy'), flow_result)
    
    # flow_result['original_data'] = movie
    
    optical_flow.make_velocity_overlay_movie(flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','velocities_for_cell_12.mp4'),arrow_scale = 0.02, arrow_boxsize = 20
                                             )
                                            #  v_max = np.max(flow_result['original_data']))

def visualise_ground_truth_displacement():
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    
    delta_x = 1.0
    delta_t = 1.0
    
    movie = movie[8:10]
    original_movie = movie.astype('float')
    blurred_movie = optical_flow.blur_movie(original_movie,3)
    
    filter_size = 7.5
    difference = blurred_movie[1,:,:] - blurred_movie[0,:,:]
    blurred_difference = skimage.filters.gaussian(difference, sigma =filter_size, preserve_range = True)
    corrected_second_frame = original_movie[1,:,:] - blurred_difference
    corrected_movie = np.copy(original_movie)
    corrected_movie[1,:,:] = corrected_second_frame
    # corrected_second_frame = blurred_movie[1,:,:] - difference
 
    measurements = pd.read_excel(os.path.join(os.path.dirname(__file__),'data','displacement_measurements.xlsx'))
    y_coordinates_start = measurements['x-position start'].values
    y_coordinates_end = measurements['x-position end'].values
    x_coordinates_start = measurements['y-position start'].values
    x_coordinates_end = measurements['y-position end'].values
    
    # movie = movie[8:10,:,:]
    
    threshold = 50
    thresholded_movie = movie.astype('float')
    thresholded_movie = thresholded_movie - threshold
    thresholded_movie[thresholded_movie<0] = 0
    thresholded_movie = thresholded_movie.astype('uint8')
    
    thresholded_blurred = optical_flow.blur_movie(thresholded_movie, smoothing_sigma = 3)
    thresholded_blurred_clahed = optical_flow.apply_clahe(thresholded_blurred, clipLimit = 1000)

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.cla()
        optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie), unit = 'pixels')
        if i == 0:
            plt.scatter(y_coordinates_start, x_coordinates_start, color = 'red', s = 1)
        else:
            plt.scatter(y_coordinates_end, x_coordinates_end, color = 'red', s = 1)
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    ani.save(os.path.join(os.path.dirname(__file__),'output','ground_truth_displacement.mp4'),dpi=300) 
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.cla()
        optical_flow.costum_imshow(thresholded_blurred_clahed[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(thresholded_blurred_clahed))
        # optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
        if i == 0:
            plt.scatter(x_coordinates_start, y_coordinates_start, color = 'red', s = 1)
        else:
            plt.scatter(x_coordinates_end, y_coordinates_end, color = 'red', s = 1)
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    ani.save(os.path.join(os.path.dirname(__file__),'output','thresholded_displacement_movie.mp4'),dpi=300) 
 
    # subtract the mean 
    # float_movie = movie.astype('float')
    # float_movie[1,:,:] *= np.mean(float_movie[0,:,:])/np.mean(float_movie[1,:,:])
    # movie = float_movie.astype('uint8')

    # flow_result = optical_flow.conduct_opencv_flow(original_movie, delta_x = delta_x, delta_t = delta_t, smoothing_sigma = None, pyr_scale = 0.5,
    flow_result = optical_flow.conduct_opencv_flow(original_movie, delta_x = delta_x, delta_t = delta_t, smoothing_sigma = None, pyr_scale = 0.5,
                                                   levels = 5, winsize = 50, iterations = 50, poly_n = 5, poly_sigma = 1.1)

    # flow_result = optical_flow.conduct_optical_flow(movie, boxsize = 25, delta_x = 1.0, delta_t = 1.0, smoothing_sigma = 3, background = None, include_remodelling = True)

    optical_flow.make_velocity_overlay_movie(flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','velocities_for_cell_12.mp4'),arrow_scale = 0.02, arrow_boxsize = 20
                                             )
    
    fig = plt.figure(figsize = (2.5,2.5),constrained_layout = True)
    plt.cla()
    optical_flow.costum_imshow(movie[0,:,:], delta_x = flow_result['delta_x'])
    plt.quiver(y_coordinates_start,x_coordinates_start, y_coordinates_end - y_coordinates_start, x_coordinates_start - x_coordinates_end, 
               color = 'blue',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.quiver(y_coordinates_start, x_coordinates_start, flow_result['v_y'][0,x_coordinates_start,y_coordinates_start], -flow_result['v_x'][0,x_coordinates_start,y_coordinates_start], 
               color = 'magenta',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','velocity_validation.pdf'),dpi = 600)
    
    relative_errors =  np.sqrt(np.power(y_coordinates_end - y_coordinates_start - flow_result['v_y'][0,x_coordinates_start,y_coordinates_start],2)
                              +np.power(x_coordinates_end - x_coordinates_start - flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],2))
    relative_errors /= np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
                              +np.power(x_coordinates_end - x_coordinates_start,2))

    print('the relative errors are')
    print(relative_errors)
    
    true_displacements = np.vstack((x_coordinates_end - x_coordinates_start,
                                    y_coordinates_end - y_coordinates_start)).transpose()
    inferred_displacements = np.vstack((flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],
                                       flow_result['v_y'][0,x_coordinates_start,y_coordinates_start])).transpose()
    print(' ')
    print('true displacements')
    print(true_displacements)
    print('inferred_displacement')
    print(inferred_displacements)

    print(' ')
    print('positions')
    print(np.vstack((x_coordinates_start,y_coordinates_start)).transpose())
    
    print('the actual displacements are')
    print(np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
           +np.power(x_coordinates_end - x_coordinates_start,2)))

    print('the inferred displacements are')
    print(np.sqrt(np.power(flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],2)
           +np.power(flow_result['v_y'][0,x_coordinates_start,y_coordinates_start],2)))

def correct_intensity_change():
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))

    movie = movie[8:10]
    original_movie = movie.astype('float')
    blurred_movie = optical_flow.blur_movie(original_movie,3)
    
    filter_size = 5
    difference = blurred_movie[1,:,:] - blurred_movie[0,:,:]
    blurred_difference = skimage.filters.gaussian(difference, sigma =filter_size, preserve_range = True)
    corrected_second_frame = blurred_movie[1,:,:] - blurred_difference
    # corrected_second_frame = blurred_movie[1,:,:] - difference
    
    plt.figure(figsize = (6.5,6.5), constrained_layout = True)

    x_values = range(blurred_movie.shape[2])
    range_start = 350
    range_end = 600
    plt.subplot(331)
    optical_flow.costum_imshow(original_movie[0,:,:],delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.axhline(450)
    plt.title('original')
    plt.subplot(332)
    plt.plot(x_values,original_movie[0,450,:],lw = 0.1)
    plt.plot(x_values,blurred_movie[0,450,:],lw = 0.3)
    plt.subplot(333)
    plt.plot(x_values[range_start:range_end],original_movie[0,450,range_start:range_end],lw = 0.1)
    plt.plot(x_values[range_start:range_end],blurred_movie[0,450,range_start:range_end],lw = 0.3)

    plt.subplot(334)
    optical_flow.costum_imshow(blurred_movie[0,:,:],delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.title('First blurred')
    plt.subplot(335)
    optical_flow.costum_imshow(difference,delta_x = 1.0, v_min = 0, v_max = np.max(difference))
    plt.colorbar()
    # optical_flow.costum_imshow(difference,delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.title('difference')
    plt.subplot(336)
    optical_flow.costum_imshow(blurred_difference,delta_x = 1.0, v_min = 0, v_max = np.max(blurred_difference))
    plt.colorbar()
    # optical_flow.costum_imshow(difference,delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.title('blurred difference')
 
    plt.title('Profiles')
    plt.subplot(337)
    optical_flow.costum_imshow(blurred_movie[1,:,:],delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.title('Second frame')
    plt.subplot(338)
    optical_flow.costum_imshow(corrected_second_frame,delta_x = 1.0, v_min = 0, v_max = np.max(movie))
    plt.title('Second frame corrected')
    plt.subplot(339)
    range_start = 350
    range_end = 450
    plt.plot(x_values[range_start:range_end], blurred_movie[0,450,range_start:range_end],lw = 0.1, label = '0')
    plt.plot(x_values[range_start:range_end], blurred_movie[1,450,range_start:range_end],lw = 0.1, label = '1')
    plt.plot(x_values[range_start:range_end], corrected_second_frame[450,range_start:range_end],lw = 0.1, label = 'corrected')
    plt.legend(loc = 'lower left')
    plt.title('Profiles')
 
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','illumination_correction.pdf'),dpi = 600)

def investigate_intensities():
    """This makes a figure with the intensity histograms of both movies"""

    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    
    plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    plt.hist(movie.flatten(),bins=255, range = (0,255))
    plt.xlabel('Intensity value')
    plt.xlim(3,120)
    plt.ylim(0,5e6)
    plt.ylabel('Number of pixels')
    
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','new_data_intensity_histgram.pdf'))
    
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(121)
        optical_flow.costum_imshow(movie[i,:,:],delta_x = 1.0, v_min = 0, v_max = np.max(movie))

        plt.subplot(122)
        plt.hist(movie[i,:,:].flatten(),bins=255, range = (0,255))
        plt.xlabel('Intensity value')
        plt.xlim(3,120)
        plt.ylim(0,2e4)
        plt.ylabel('Number of pixels')
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','animated_intensities.mp4'),dpi=300) 
 
    print('Unique intensity values are')
    print(np.unique(movie))
 
def make_movie_wo_background(threshold = 50):
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))

    thresholded_movie = movie.astype('float')
    thresholded_movie = thresholded_movie - threshold
    thresholded_movie[thresholded_movie<0] = 0
    thresholded_movie = thresholded_movie.astype('uint8')

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(121)
        optical_flow.costum_imshow(movie[i,:,:],autoscale = False, delta_x = 1.0)
        plt.subplot(122)
        optical_flow.costum_imshow(thresholded_movie[i,:,:],autoscale = False, delta_x = 1.0)
    # ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','new_data_background_removed.mp4'),dpi=300) 

def make_thresholded_movie( threshold = 17.5, sigma = 1.0, clahe = None, adaptive = False):
    """This function makes a movie with both channels, and in which pixel values below a certain intensity are coloured in green."""
    
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))

    if clahe is not None:
        movie = optical_flow.apply_clahe(movie, clipLimit = clahe)
        movie /= np.max(movie)
        movie*=255.0
        clahe_string = '_w_clahe_'
    else:
        clahe_string = ''
    
    # movie_blurred = optical_flow.blur_movie(rho_movie, smoothing_sigma = rho_sigma)
    # actin_movie_blurred = optical_flow.blur_movie(actin_movie, smoothing_sigma = actin_sigma)
    
    if adaptive:
        mask = optical_flow.apply_adaptive_threshold(movie, window_size = 151, threshold = -5)
    else:
        mask = movie<threshold
    
    thresholded = np.zeros((movie.shape[0],movie.shape[1],movie.shape[2], 3), dtype = 'int')
    
    thresholded[np.logical_not(mask),1] = 255-movie[np.logical_not(mask)]
    thresholded[mask,0] = 255-movie[mask]
    thresholded[mask,1] = 255-movie[mask]
    thresholded[mask,2] = 255-movie[mask]

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        # plt.cla()
        optical_flow.costum_imshow(thresholded[i,:,:,:],autoscale = False, cmap = None, delta_x = 1.0)
    # ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    ani = FuncAnimation(fig, animate, frames=3)
    ani.save(os.path.join(os.path.dirname(__file__),'output','data_movie_thresholded_treshold_' + "{:.2f}".format(threshold) + 
                          "{:.2f}".format(sigma)+ '_' +  clahe_string + '.mp4'),dpi=300) 

def apply_variational_optical_flow(speed_regularisation=1e4, remodelling_regularisation=1e2):

    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    delta_x = 1
    delta_t = 1

    # movie = movie[3:5,150:800,280:550]
    # movie = movie[3:5,250:900,500:800]
    # movie = movie[8:10,300:750,280:600]
    movie = movie[8:10,:,:]
    blur = 1.5
    # print(movie.flatten().shape)
    # fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    # def animate(i): 
        # plt.cla()
        # optical_flow.costum_imshow(movie[i,:,:],delta_x = delta_x, v_min = 0, v_max = np.max(movie))
    # ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    # ani = FuncAnimation(fig, animate, frames=3)
    # ani.save(os.path.join(os.path.dirname(__file__),'output','pretty_real_movie_smaller.mp4'),dpi=300) 
 
    # iterations = 10000
    # iteration_stepsize = 500
    # movie = movie[:10,:,:]
    result = optical_flow.variational_optical_flow(movie,
                                                           delta_x = delta_x,
                                                           delta_t = delta_t,
                                                           speed_alpha=speed_regularisation,
                                                           remodelling_alpha=remodelling_regularisation,
                                                           initial_v_x =0.07,
                                                           initial_v_y =0.07,
                                                           initial_remodelling=10,
    # )
                                                           smoothing_sigma = blur,
                                                        #    use_direct_solver = True)
                                                           use_direct_solver = False)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','cell_12_test_result_' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '_' + str(blur) + '.npy'), result)
    # result = np.load(os.path.join(os.path.dirname(__file__),'output','real_data_test_result_' + str(speed_regularisation) + '_'
                                                        #   + str(remodelling_regularisation) + '_' + str(blur) + '.npy'),allow_pickle='TRUE').item()

    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'cell_12_test_' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '_' + str(blur) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.05,
                                             arrow_boxsize = 20)
                                            #  arrow_width = 0.005)
                                            #  arrow_color = 'lime')

    optical_flow.make_joint_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                             'cell_12_test' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '_' + str(blur) + '_joint_result.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.05,
                                             arrow_boxsize = 20)
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
 
def perform_tuning_variation():
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    delta_x = 1
    delta_t = 1

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
    # movie = movie[8:10,1:50,1:50]
    movie = movie[8:10,:,:]
 
    # result_for_plotting = optical_flow.vary_regularisation(movie, speed_alpha_values = np.logspace(3,8,15),
                                                        #    remodelling_alpha_values = np.logspace(-1,8,20),
    result_for_plotting = optical_flow.vary_regularisation(movie, speed_alpha_values = np.logspace(3,8,3),
                                                           remodelling_alpha_values = np.logspace(-1,8,3),
                                                           filename = os.path.join(os.path.dirname(__file__), 'output',
                                                                                   'new_data_regularisation_variation'),
                                                           smoothing_sigma = 3)

    # result_for_plotting = np.load(os.path.join(os.path.dirname(__file__),'output','real_data_regularisation_variation.npy'),allow_pickle='TRUE').item()
    optical_flow.plot_regularisation_variation(result_for_plotting, os.path.join(os.path.dirname(__file__), 'output',
                                                                                   'new_data_regularisation_variation.pdf'),
                                                                                   use_log_axes = True,
                                                                                   use_log_colorbar = False)
 
def show_PIV_data():
    loading_path = os.path.join(os.path.dirname(__file__), 'data', 'PIVlab_cell_12.mat')
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))

    PIV_result = scipy.io.loadmat(loading_path)
    x_locations, y_locations, v_x, v_y = optical_flow.convert_PIV_result(PIV_result)
    
    # import pdb; pdb.set_trace()
    
    fig = plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    optical_flow.costum_imshow(movie[8,:,:], delta_x = 1.0, autoscale=False)
    plt.quiver( x_locations[0,:,:],y_locations[0,:,:], v_x[0,:,:],-v_y[0,:,:], color = 'magenta',headwidth=5, scale = 1/0.005)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    saving_name = os.path.join(os.path.dirname(__file__),'output', 'piv_visualisation_cell_12_frame_9.pdf')
    plt.savefig(saving_name,dpi=600) 
    
    # Create a meshgrid for the whole image
    x_range = np.arange(0, movie.shape[1])
    y_range = np.arange(0, movie.shape[2])
    X, Y = np.meshgrid(x_range, y_range)

    nan_mask = np.logical_and(~np.isnan(v_x),~np.isnan(v_y))

    # Upsample x-velocities and y-velocities using quadratic interpolation
    x_velocities_upsampled = scipy.interpolate.griddata((x_locations[nan_mask].flatten(), y_locations[nan_mask].flatten()), 
                                  v_x[nan_mask].flatten(), 
                                  (X, Y), 
                                  method='cubic')
    y_velocities_upsampled = scipy.interpolate.griddata((x_locations[nan_mask].flatten(), y_locations[nan_mask].flatten()), 
                                  v_y[nan_mask].flatten(), 
                                  (X, Y), 
                                  method='cubic')
     
    x_velocities_upsampled = np.array([x_velocities_upsampled.reshape(x_range.size, y_range.size)])
    y_velocities_upsampled = np.array([y_velocities_upsampled.reshape(x_range.size, y_range.size)])
    speed = np.sqrt(x_velocities_upsampled**2 + y_velocities_upsampled**2)
    
    blurred_movie = optical_flow.blur_movie(movie,3)

    x_velocities_upsampled[0,blurred_movie[8,:,:]<10.0] = 0.0
    y_velocities_upsampled[0,blurred_movie[8,:,:]<10.0] = 0.0
    x_velocities_upsampled[speed> 7] = 0.0
    y_velocities_upsampled[speed> 7] = 0.0
    flow_result = dict()
    flow_result['v_x'] = x_velocities_upsampled
    flow_result['v_y'] = y_velocities_upsampled
    flow_result['delta_x'] = 1.0
    flow_result['delta_t'] = 1.0
    flow_result['original_data'] = movie[8:10,:,:]
    
    optical_flow.make_velocity_overlay_movie(flow_result, 
                                             os.path.join(os.path.dirname(__file__),'output','velocities_for_cell_12_PIV.mp4'),arrow_scale = 0.02, arrow_boxsize = 20
                                             )

    measurements = pd.read_excel(os.path.join(os.path.dirname(__file__),'data','displacement_measurements.xlsx'))
    y_coordinates_start = measurements['x-position start'].values
    y_coordinates_end = measurements['x-position end'].values
    x_coordinates_start = measurements['y-position start'].values
    x_coordinates_end = measurements['y-position end'].values
 
    fig = plt.figure(figsize = (2.5,2.5),constrained_layout = True)
    plt.cla()
    optical_flow.costum_imshow(movie[8,:,:], delta_x = flow_result['delta_x'])
    plt.quiver(y_coordinates_start,x_coordinates_start, y_coordinates_end - y_coordinates_start, x_coordinates_start - x_coordinates_end, 
               color = 'blue',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.quiver(y_coordinates_start, x_coordinates_start, flow_result['v_y'][0,x_coordinates_start,y_coordinates_start], -flow_result['v_x'][0,x_coordinates_start,y_coordinates_start], 
               color = 'magenta',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','velocity_validation_PIV.pdf'),dpi = 600)
    
    relative_errors =  np.sqrt(np.power(y_coordinates_end - y_coordinates_start - flow_result['v_y'][0,x_coordinates_start,y_coordinates_start],2)
                              +np.power(x_coordinates_end - x_coordinates_start - flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],2))
    relative_errors /= np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
                              +np.power(x_coordinates_end - x_coordinates_start,2))

    print('the relative errors are')
    print(relative_errors)
    
    true_displacements = np.vstack((x_coordinates_end - x_coordinates_start,
                                    y_coordinates_end - y_coordinates_start)).transpose()
    inferred_displacements = np.vstack((flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],
                                       flow_result['v_y'][0,x_coordinates_start,y_coordinates_start])).transpose()
    print(' ')
    print('true displacements')
    print(true_displacements)
    print('inferred_displacement')
    print(inferred_displacements)

    print(' ')
    print('positions')
    print(np.vstack((x_coordinates_start,y_coordinates_start)).transpose())
    
    print('the actual displacements are')
    print(np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
           +np.power(x_coordinates_end - x_coordinates_start,2)))

    print('the inferred displacements are')
    print(np.sqrt(np.power(flow_result['v_x'][0,x_coordinates_start,y_coordinates_start],2)
           +np.power(flow_result['v_y'][0,x_coordinates_start,y_coordinates_start],2)))

def compare_PIV_and_optical_flow():
    loading_path = os.path.join(os.path.dirname(__file__), 'data','PIVlab_cell_12_all.mat')
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))

    PIV_result = scipy.io.loadmat(loading_path)
    PIV_flow_result = optical_flow.convert_PIV_result(PIV_result, movie)
    np.save(os.path.join(os.path.dirname(__file__),'output','PIV_optical_flow_result.npy'), PIV_flow_result)
    # PIV_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','PIV_optical_flow_result.npy'),allow_pickle='TRUE').item()

    optical_flow.filter_PIV_flow_result(PIV_flow_result,intensity_threshold = 10)

    # opencv_flow_result = optical_flow.conduct_opencv_flow(movie, delta_x = 1.0, delta_t = 1.0, smoothing_sigma = None, pyr_scale = 0.5,
                                                            #  levels = 5, winsize = 50, iterations = 50, poly_n = 5, poly_sigma = 1.1)
    # np.save(os.path.join(os.path.dirname(__file__),'output','opencv_flow_result.npy'), opencv_flow_result)
    opencv_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','opencv_flow_result.npy'),allow_pickle='TRUE').item()
 
    x_positions, y_positions, v_x_PIV, v_y_PIV = optical_flow.subsample_velocities_for_visualisation(PIV_flow_result, arrow_boxsize = 15)
    x_positions, y_positions, v_x_opencv, v_y_opencv = optical_flow.subsample_velocities_for_visualisation(opencv_flow_result, arrow_boxsize = 15)

    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(121)
        plt.title('PIV')
        optical_flow.costum_imshow(movie[i,:,:],delta_x = 1.0, unit = 'pixels')
        plt.quiver(y_positions, x_positions, v_y_PIV[i,:,:], -v_x_PIV[0,:,:], color = 'magenta',headwidth=5, scale = None)
        plt.subplot(122)
        plt.title('OpenCV')
        optical_flow.costum_imshow(movie[i,:,:],delta_x = 1.0, unit = 'pixels')
        plt.quiver(y_positions, x_positions, v_y_opencv[i,:,:], -v_x_opencv[0,:,:], color = 'magenta',headwidth=5, scale = None)
    ani = FuncAnimation(fig, animate, frames=movie.shape[0]-1)
    # ani = FuncAnimation(fig, animate, frames=2)

    ani.save(os.path.join(os.path.dirname(__file__),'output','PIV_joint_flow_visualisation.mp4'),dpi=600) 

def compare_ground_truth_displacement():
    '''will only work if the function above has already been run because we are loading their results back in'''
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    
    delta_x = 1.0
    delta_t = 1.0
    
    measurements = pd.read_excel(os.path.join(os.path.dirname(__file__),'data','displacement_measurements.xlsx'))
    y_coordinates_start = measurements['x-position start'].values
    y_coordinates_end = measurements['x-position end'].values
    x_coordinates_start = measurements['y-position start'].values
    x_coordinates_end = measurements['y-position end'].values
    
    PIV_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','PIV_optical_flow_result.npy'),allow_pickle='TRUE').item()
    optical_flow.filter_PIV_flow_result(PIV_flow_result,intensity_threshold = 10)
    opencv_flow_result = np.load(os.path.join(os.path.dirname(__file__),'output','opencv_flow_result.npy'),allow_pickle='TRUE').item()

    relative_errors_PIV =  np.sqrt(np.power(y_coordinates_end - y_coordinates_start - PIV_flow_result['v_y'][8,x_coordinates_start,y_coordinates_start],2)
                              +np.power(x_coordinates_end - x_coordinates_start - PIV_flow_result['v_x'][8,x_coordinates_start,y_coordinates_start],2))
    relative_errors_PIV /= np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
                              +np.power(x_coordinates_end - x_coordinates_start,2))

    relative_errors_opencv =  np.sqrt(np.power(y_coordinates_end - y_coordinates_start - opencv_flow_result['v_y'][8,x_coordinates_start,y_coordinates_start],2)
                              +np.power(x_coordinates_end - x_coordinates_start - opencv_flow_result['v_x'][8,x_coordinates_start,y_coordinates_start],2))
    relative_errors_opencv /= np.sqrt(np.power(y_coordinates_end - y_coordinates_start,2)
                              +np.power(x_coordinates_end - x_coordinates_start,2))
   
    fig = plt.figure(figsize = (6.5,2.5),constrained_layout = True)

    plt.subplot(131)
    optical_flow.costum_imshow(movie[0,:,:], delta_x = PIV_flow_result['delta_x'],unit = 'pixels')
    plt.quiver(y_coordinates_start,x_coordinates_start, y_coordinates_end - y_coordinates_start, x_coordinates_start - x_coordinates_end, 
               color = 'blue',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.quiver(y_coordinates_start, x_coordinates_start, PIV_flow_result['v_y'][8,x_coordinates_start,y_coordinates_start], -PIV_flow_result['v_x'][8,x_coordinates_start,y_coordinates_start], 
               color = 'magenta',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.title('PIV')

    plt.subplot(132)
    optical_flow.costum_imshow(movie[0,:,:], delta_x = opencv_flow_result['delta_x'],unit = 'pixels')
    plt.quiver(y_coordinates_start,x_coordinates_start, y_coordinates_end - y_coordinates_start, x_coordinates_start - x_coordinates_end, 
               color = 'blue',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.quiver(y_coordinates_start, x_coordinates_start, PIV_flow_result['v_y'][8,x_coordinates_start,y_coordinates_start], -opencv_flow_result['v_x'][8,x_coordinates_start,y_coordinates_start], 
               color = 'magenta',headwidth=5, scale = 1.0/0.05, width = None)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
    plt.title('OpenCV')
    
    plt.subplot(133)
    indices = np.arange(5)
    bar_width = 0.35
    # Create the bar chart
    plt.bar(indices - bar_width / 2, relative_errors_PIV, bar_width, label='PIV')
    plt.bar(indices + bar_width / 2, relative_errors_opencv, bar_width, label='OpenCV')
    plt.ylabel('relative error')
    plt.legend()
    plt.xlabel('data point')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','ground_truth_comparison.pdf'),dpi = 600)

    PIV_flow_result['speed'][np.isnan(PIV_flow_result['speed'])] = 0.0
    opencv_flow_result['speed'][np.isnan(opencv_flow_result['speed'])] = 0.0
    PIV_flow_result_for_correlation = PIV_flow_result['speed'][np.logical_and(PIV_flow_result['speed']>0.1,opencv_flow_result['speed']>0.1)]
    opencv_flow_result_for_correlation = opencv_flow_result['speed'][np.logical_and(PIV_flow_result['speed']>0.1,opencv_flow_result['speed']>0.1)]
    plt.figure(figsize = (3.5, 2.5), constrained_layout = True)
    hist, xedges, yedges, _ = plt.hist2d(PIV_flow_result_for_correlation.flatten(), opencv_flow_result_for_correlation.flatten(), bins = (50,50))
    # hist, xedges, yedges, _ = plt.hist2d(PIV_flow_result_for_correlation.flatten(), opencv_flow_result_for_correlation.flatten(), bins = (5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.xticks([1,2,3,4])
    plt.yticks([1,2,3,4])

    # plt.xlim(0.1,)
    # plt.ylim(0.1,)
    plt.xlabel('PIV speed [pxl/frame]')
    plt.ylabel('opencv speed [pxl/frame]')
    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('number of boxes')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','speed_correlation_PIV.png'),dpi=300) 

    scalar_products = (opencv_flow_result['v_x']*PIV_flow_result['v_x'] +
                       opencv_flow_result['v_y']*PIV_flow_result['v_y'])
    
    cos_values = scalar_products/(opencv_flow_result['speed']*PIV_flow_result['speed'])
    theta_values = np.arccos(cos_values)
    cos_values = cos_values[np.logical_and(opencv_flow_result['speed']>0.1,PIV_flow_result['speed']>0.1)]
    theta_values = theta_values[np.logical_and(opencv_flow_result['speed']>0.5,PIV_flow_result['speed']>0.5)]
    # weights = weights[np.logical_and(speed_actin_unfiltered>0.0,speed_rho_unfiltered>0.0)]

    plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    ax = plt.subplot(projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_thetalim((0.0,np.pi))
    ax.set_xticks(np.linspace(0, np.pi, 5))  # Set tick positions
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°'])  # Set tick labels
    ax.hist(theta_values.flatten(), bins = 20)
    ax.text(-0.05, 0.5, 'Number of pixels', rotation='vertical', va='center', ha='center', transform=ax.transAxes)
    plt.title('Angle between\nPIV and OpenCV result')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','polar_histogram_PIV.pdf'),dpi=600) 
    
    plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    plt.hist(theta_values.flatten(), bins = 20)
    plt.xticks(np.linspace(0, np.pi, 5))  # Set tick positions
    plt.gca().set_xticklabels(['0°', '45°', '90°', '135°', '180°'])  # Set tick labels
    plt.ylabel('Number of pixels')
    plt.xlabel('Angle between\nPIV and OpenCV result')
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','angle_histogram_PIV.pdf'),dpi=600) 

def try_downsampled_image():
    path_to_movie = os.path.join(os.path.dirname(__file__),'data','12_grayscale')
    movie = np.array(read_image_sequence(path_to_movie))
    movie = movie[8:10].astype('float')
    # movie = optical_flow.blur_movie(movie,3)
    
    resolution = 200
    downsampled_movie = np.zeros((movie.shape[0],resolution,resolution))
    for frame_index in range(movie.shape[0]):
        this_frame = movie[frame_index,:,:]
        # this_downsampled_frame = cv2.resize(this_frame,dsize = (50,50), interpolation = cv2.INTER_CUBIC)
        this_downsampled_frame = cv2.resize(this_frame,dsize = (resolution,resolution), interpolation = cv2.INTER_AREA)
        downsampled_movie[frame_index,:,:] = this_downsampled_frame
    
    # animate the downsampled movie
    fig = plt.figure(figsize = (4.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.cla()
        optical_flow.costum_imshow(downsampled_movie[i,:,:],delta_x = 1, v_min = 0, v_max = np.max(movie))
    ani = FuncAnimation(fig, animate, frames=movie.shape[0])
    ani.save(os.path.join(os.path.dirname(__file__),'output','downsampled_movie.mp4'),dpi=300) 
    
    tifffile.imsave(os.path.join(os.path.dirname(__file__),'output','downsampled.tiff'), downsampled_movie)

    speed_regularisation = 500
    remodelling_regularisation = 1
    result = optical_flow.variational_optical_flow(downsampled_movie,
                                                           delta_x = 1,
                                                           delta_t = 1,
                                                           speed_alpha=speed_regularisation,
                                                           remodelling_alpha=remodelling_regularisation,
                                                           initial_v_x =0.07,
                                                           initial_v_y =0.07,
                                                           initial_remodelling=10,
    # )
                                                           smoothing_sigma = 1.0,
                                                        #    use_direct_solver = True)
                                                           use_direct_solver = True)
    
    np.save(os.path.join(os.path.dirname(__file__),'output','cell_12_test_result_downsampled_' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '.npy'), result)
    # result = np.load(os.path.join(os.path.dirname(__file__),'output','real_data_test_result_' + str(speed_regularisation) + '_'
                                                        #   + str(remodelling_regularisation) + '_' + str(blur) + '.npy'),allow_pickle='TRUE').item()

    optical_flow.make_velocity_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                                          'cell_12_test_downsampled' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.1,
                                             arrow_boxsize = 5)
                                            #  arrow_width = 0.005)
                                            #  arrow_color = 'lime')

    optical_flow.make_joint_overlay_movie(result, 
                                             os.path.join(os.path.dirname(__file__),'output',
                                             'cell_12_test_downsampled_' + str(speed_regularisation) + '_'
                                                          + str(remodelling_regularisation) + '_joint_result.mp4'), 
                                             autoscale = True,
                                             arrow_scale = 0.1,
                                             arrow_boxsize = 5)
                                            #  arrow_width = 0.005)



if __name__ == '__main__':

    # functions I used in the presentation:
    # visualise_ground_truth_displacement()
    # compare_PIV_and_optical_flow()
    # compare_ground_truth_displacement()

    # other functions I played around with
    # conduct_dense_optical_flow_one_frame()
    # perform_tuning_variation()
    # investigate_intensities()
    # make_thresholded_movie(threshold = 100)
    # make_movie_wo_background(threshold = 40)
    # correct_intensity_change()
    
    try_downsampled_image()
    # apply_variational_optical_flow(speed_regularisation=2e3, remodelling_regularisation=1)

    # show_PIV_data()