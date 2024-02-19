import numpy as np
from numba import jit
from numba import njit
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('figure', titlesize=10)     # fontsize of the axes title
import matplotlib.ticker

from matplotlib.animation import FuncAnimation
import skimage.filters
import cv2

@jit(nopython=True, error_model = "numpy")
def conduct_optical_flow_jit(movie, box_size = 15, delta_x = 1.0, delta_t = 1.0, include_remodelling = False):
    """This is a helper method for conduct_optical_flow below. It ensures that the actual
       optical flow calculations are conducted with numba just-in-time compiled code.
       
       Resulting data of v_x, v_y and speed have one less frame than the movie, and the ith frame
       was calculated from the difference between frame i-1 and frame i
       
    Parameters :
    ------------

    movie : np.array
        the movie we wish to analyse
    
    boxsize : int
        the boxsize for the optical flow algorithm. If an even number is provided, the next smallest uneven number will be used.

    delta_x : float
        the size of one pixel in the movie. We assume pixel size is identical in the x and y directions

    delta_t : float
        the time interval between frames in the movie. Defaults to 1.0. 
        
    include_remodelling : bool
        if True return the net remodelling as well as v_x and v_y
        
    Returns :
    ---------
    
    v_x : np array
        The calculated x-velocities at each pixel

    v_y : np array
        The calculated y-velocities at each pixel

    speed : np array
        The speed at each pixel
    
    net_remodelling : np array
        The net remodelling rate at each location. If include_remodelling is False, an array of zeros will be returned
    """
    
    # movie = np.array(movie, dtype = np.float64)
    movie = movie.astype(np.float64)
    first_frame = movie[0,:,:]
    dIdx = np.zeros_like(first_frame)
    dIdy = np.zeros_like(first_frame)
    delta_t_data = delta_t
    delta_t = 1
    number_of_frames = movie.shape[0]
    Xpixels=movie.shape[1]#Number of X pixels
    Ypixels=movie.shape[2]#Number of Y pixels
    all_v_x = np.zeros((number_of_frames-1,Xpixels,Ypixels))
    all_v_y = np.zeros((number_of_frames-1,Xpixels,Ypixels))
    all_net_remodelling = np.zeros((number_of_frames-1,Xpixels,Ypixels))
    #get average speed of each frame:|v|=np.sqrt(Vx**2+Vy**2), np.mean(V)
    all_speed= np.zeros((number_of_frames-1,Xpixels,Ypixels))
    
    print('conducting optical flow now')
    for frame_index in range(1,movie.shape[0]):
        print('current_frame: ' + str(frame_index))
        current_frame = movie[frame_index]
        previous_frame = movie[frame_index -1]
            
        dIdx_calc = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/4
        dIdy_calc = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/4
        
        dIdx[1:-1,1:-1] = dIdx_calc
        dIdy[1:-1,1:-1] = dIdy_calc

        delta_I = current_frame-previous_frame
       
        v_x = all_v_x[frame_index-1,:,:]
        v_y = all_v_y[frame_index-1,:,:]
        net_remodelling = all_net_remodelling[frame_index-1,:,:]
        
        speed= all_speed[frame_index-1,:,:]
        
        for pixel_index_x in range(movie.shape[1]):
            #from0,...Nb-1
            for pixel_index_y in range(movie.shape[2]):#better chose odd number box_size to make sure it's symmetrical
                x_begining = (max(pixel_index_x-int(box_size/2),0))#e.g box_size=11,half is 5 and pixel in center
                x_end = (min(pixel_index_x+int(box_size/2)+1,movie.shape[1]))#e.g box_size=11,pixel=20 then it's 26 but stop at 25
                y_beginning = (max(pixel_index_y-int(box_size/2),0))
                y_end = (min(pixel_index_y+int(box_size/2)+1,movie.shape[1]))
                
                local_delta_I = delta_I[x_begining:x_end,y_beginning:y_end]
                #print("local"+str(frame_index)+"+"+str(pixel_index_x))
                
                local_dIdx = dIdx[x_begining:x_end,y_beginning:y_end]
                local_dIdy = dIdy[x_begining:x_end,y_beginning:y_end]
                
                sum1 = np.sum(local_delta_I*local_dIdx)
                sum2 = np.sum(local_delta_I*local_dIdy)

                if not include_remodelling:
                    A = np.sum((local_dIdx)**2)
                    B = np.sum(local_dIdx*local_dIdy)
                    C = np.sum((local_dIdy)**2)
                    Vx =(-C*sum1+ B*sum2)/(delta_t*(A*C-B**2))
                    Vy =(-A*sum2+ B*sum1)/(delta_t*(A*C-B**2))
                    Vspeed=np.sqrt(Vx**2+Vy**2)

                    v_x[pixel_index_x,pixel_index_y] = Vx
                    v_y[pixel_index_x,pixel_index_y] = Vy
                
                    speed[pixel_index_x,pixel_index_y] = Vspeed
                else:
                    A = np.sum((local_dIdx)**2)
                    B = np.sum(local_dIdx*local_dIdy)
                    C = np.sum(local_dIdx)
                    D = np.sum((local_dIdy)**2)
                    E = np.sum(local_dIdy)
                    sum3 = np.sum(local_delta_I)
                    
                    total_boxsize = box_size*box_size
                    this_sumde = delta_t*(total_boxsize*A*D-A*E**2-total_boxsize*B**2-C**2*D+2*B*C*E)
                    if this_sumde == 0.0:
                        Vx = np.nan
                        Vy = np.nan
                        this_gamma = np.nan
                    else:
                        Vx = ((E**2-total_boxsize*D)*sum1+(total_boxsize*B-C*E)*sum2+(C*D-B*E)*sum3)/this_sumde
                        Vy = ((total_boxsize*B-C*E)*sum1+(C**2-total_boxsize*A)*sum2+(A*E-B*C)*sum3)/this_sumde
                        v_x[pixel_index_x,pixel_index_y] = Vx
                        v_y[pixel_index_x,pixel_index_y] = Vy
                        this_gamma = -((B*E-C*D)*sum1+(B*C-A*E)*sum2+(A*D-B**2)*sum3)/this_sumde#gamma add"-"20230206
                        net_remodelling[pixel_index_x,pixel_index_y] = this_gamma
                        
        v_x*= delta_x/delta_t_data
        v_y*= delta_x/delta_t_data
        speed*=delta_x/delta_t_data
        
    return all_v_x, all_v_y, all_speed, all_net_remodelling

def conduct_optical_flow(movie, boxsize = 15, delta_x = 1.0, delta_t = 1.0, smoothing_sigma = None, background = None, include_remodelling = False):
    """Conduct optical flow as in Vig et al. Biophysical Journal 110, 1469–1475, 2016.
    
    Parameters:
    -----------

    movie : np.array
        the movie we wish to analyse
    
    boxsize : int
        the boxsize for the optical flow algorithm. If an even number is provided, the next smallest uneven number will be used.

    delta_x : float
        the size of one pixel in the movie. We assume pixel size is identical in the x and y directions

    delta_t : float
        the time interval between frames in the movie. Defaults to 1.0.
        
    smoothing_sigma : float or None
        If the value None is provided, no smoothing will be applied. Otherwise a gaussian blur with this sigma value
        will be applied to the movie before optical flow is conducted.

    background : float or None
        If the value None is provided, no background will be subtracted. Otherwise, the background level will be subtracted before optical flow is conducted.

    include_remodelling : bool
        if True return the net remodelling as well as v_x and v_y

    Returns:
    --------

    result : dict
        A dictionary containing the results of optical flow calculations, as well as arrays for the orignal and blurred data.
        The keys are: v_x, v_y, speed, original_data, blurred_data, delta_x, delta_t
    """
    
    if background is not None:
        movie_for_thresholding = blur_movie(movie, smoothing_sigma=10)
        movie_to_analyse = np.zeros_like(movie_for_thresholding)
        movie_to_analyse[movie_for_thresholding > background] = movie[movie_for_thresholding>background]- background 
    else: 
        movie_to_analyse = movie

    if smoothing_sigma is not None:
        movie_to_analyse = blur_movie(movie_to_analyse, smoothing_sigma=smoothing_sigma)

    all_v_x, all_v_y, all_speed, net_remodelling = conduct_optical_flow_jit(movie_to_analyse, boxsize, delta_x, delta_t, include_remodelling)
    result = dict()
    result['v_x'] = all_v_x
    result['v_y'] = all_v_y
    result['speed'] = all_speed
    result['original_data'] = movie
    result['delta_x'] = delta_x
    result['delta_t'] = delta_t
    result['blurred_data'] = movie_to_analyse
    
    if include_remodelling:
        result['net_remodelling'] = net_remodelling

    return result

def conduct_opencv_flow(movie, delta_x = 1.0, delta_t = 1.0, smoothing_sigma = None):
    """Conduct optical flow using the opencv library.
    
    Parameters:
    -----------

    movie : np.array
        the movie we wish to analyse
    
    delta_x : float
        the size of one pixel in the movie. We assume pixel size is identical in the x and y directions

    delta_t : float
        the time interval between frames in the movie. Defaults to 1.0.
        
    smoothing_sigma : float or None
        If the value None is provided, no smoothing will be applied. Otherwise a gaussian blur with this sigma value
        will be applied to the movie before optical flow is conducted.

    Returns:
    --------

    result : dict
        A dictionary containing the results of optical flow calculations, as well as arrays for the orignal and blurred data.
        The keys are: v_x, v_y, speed, original_data, blurred_data, delta_x, delta_t
    """
    
    if smoothing_sigma is not None:
        movie_to_analyse = blur_movie(movie, smoothing_sigma=smoothing_sigma)
    else:
        movie_to_analyse = movie

    v_x = np.zeros((movie.shape[0]-1, movie.shape[1], movie.shape[2]))
    v_y = np.zeros((movie.shape[0]-1, movie.shape[1], movie.shape[2]))
    this_result = None
    for frame_index in range(movie.shape[0]-1):
        this_result = cv2.calcOpticalFlowFarneback(movie_to_analyse[frame_index,:,:], movie[frame_index + 1,:,:], this_result, 0.5, 
                                                   levels = 5, winsize = 10, iterations = 40, poly_n = 5, poly_sigma = 10, flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        v_x[frame_index,:,:] = this_result[:,:,0]
        v_y[frame_index,:,:] = this_result[:,:,1]
        
        this_result = - this_result
        
    v_x*= delta_x/delta_t
    v_y*= delta_x/delta_t
    
    speed = np.sqrt(v_x**2 + v_y**2)

    flow_result = dict()
    flow_result['v_x'] = v_x
    flow_result['v_y'] = v_y
    flow_result['speed'] = speed
    flow_result['original_data'] = movie_to_analyse
    flow_result['delta_x'] = delta_x
    flow_result['delta_t'] = delta_t
 
    return flow_result


def blur_movie(movie, smoothing_sigma):
    """Blur a movie with the given sigma value.
    
    Parameters :
    ------------
    
    movie : np array
        The movie to be blurred. Needs to be a multi-frame single-channel movie 
        
    smoothing_sigma : float
        The sigma value to be used in the Gaussian blur
    
    Returns :
    ---------
    
    blurred_movie : np array
        The blurred movie
    """
    blurred_movie = np.zeros_like(movie, dtype ='double')
    for index in range(movie.shape[0]):
        this_frame = movie[index,:,:]
        this_blurred_image = skimage.filters.gaussian(this_frame, sigma =smoothing_sigma, preserve_range = True)
        blurred_movie[index,:,:] = this_blurred_image
    
    return blurred_movie

def apply_adaptive_threshold(movie, window_size = 51, threshold = 0.0):
    """Apply an opencv adaptive threshold to a video.
    
    Parameters :
    ------------
    
    movie : np array
        The movie to be blurred. Needs to be a multi-frame single-channel movie 
        
    window_size : int
        the blocksize of used in the cv2 adaptive threshold method

    threshold : float
        The the opencv adaptive threshold parameter

    Returns :
    ---------
    
    thresholded_movie : np array, bool type
        The movie after adaptive threshold application on each image
    """
    thresholded_movie = np.zeros_like(movie, dtype ='double')
    transformed_movie = np.array(movie/np.max(movie)*255.0, dtype = 'uint8')
    for index in range(movie.shape[0]):
        this_frame = transformed_movie[index,:,:]
        this_thresholded_frame = cv2.adaptiveThreshold(this_frame,1.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size, threshold)
        thresholded_movie[index,:,:] = this_thresholded_frame
        
    thresholded_movie = thresholded_movie == 1.0
        
    return thresholded_movie

def apply_clahe(movie, clipLimit = 50000, tile_number = 10):
    """Apply an opencv clahe to a video.
    
    Parameters :
    ------------
    
    movie : np array
        The movie to be blurred. Needs to be a multi-frame single-channel movie 
        
    clipLimit : float
        The clipLimit argument of the opencv Clahe filter. Larger numbers mean more contrast, lower numbers
        will keep the image closer to the original.

    tile_number : int
        number of tiles that is used in x-direction. The number of tiles in y-direction is scaled by the y/x aspect ratio
        so that tiles are approximately square.
        
    Returns :
    ---------
    
    clahed_movie : np array
        The movie after clahe application on each image
    """
    clahed_movie = np.zeros_like(movie, dtype ='double')
    converted_movie = movie.astype(dtype = np.uint16)
    aspect_ratio = movie.shape[2]/ movie.shape[1]
    clahe_filter = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile_number,round(tile_number*aspect_ratio)))
    for index in range(movie.shape[0]):
        this_frame = converted_movie[index,:,:]
        this_clahed_image = clahe_filter.apply(this_frame)
        clahed_movie[index,:,:] = this_clahed_image
        
    print(np.max(clahed_movie))
    
    return clahed_movie

@jit(nopython = True)
def make_fake_data_frame(x_position, y_position, sigma = 1.0, width = 20.0, include_noise = False, dimension = 1000):
    """This is a helper function for making in silico data.
    It draws a gaussian hat of the form e^[-1/sigma^2*(delta_x^2 + delta_y^2) around the given
    x and y positoins
    
    Parameters:
    -----------

    x_position : float
        the position of the hat.
        
    y_position :float
        the y position of the hat
        
    sigma : float
        the radius of the hat, see formula above
        
    x_width : float
        the x width of the domain. The domain is assumed to be square, i.e the height will be the same
        
    include_noise : bool
        whether to include random noise in the image. If true, 1e-7 is added to each pixel
        
    dimension : int
        The size of the image in pixels

    Returns:
    --------

    frame : np.array
        an image of the drawn hat

    delta_x : float
        the size of one pixel (measured in the same units as x_position, y_position, and sigma)  
    """
    x = np.linspace(0,width,dimension)#(0,10,100)0-10 100 samples uniform,np.linespace(start,stop,number)
    y = np.linspace(0,width,dimension)
    frame = np.zeros((dimension,dimension))
    for x_index, x_value in enumerate(x):
        for y_index,y_value in enumerate(y):
            frame[x_index, y_index] = np.exp((-(x_value-x_position)**2 - (y_value-y_position)**2)/sigma**2)
    delta_x = x[1]-x[0]
    if include_noise:
        frame += np.random.rand(dimension, dimension)*0.0000001
        frame = np.abs(frame)
 
    return frame, delta_x

# @jit(nopython = True)
@njit
def liu_shen_optical_flow_jit(movie,
                             delta_x = 1.0,
                             delta_t = 1.0,
                             alpha=100,
                             initial_v_x=np.zeros((10,10),dtype =float),
                             initial_v_y= np.zeros((10,10),dtype =float),
                             initial_remodelling=np.zeros((10,10),dtype =float),
                             iterations = 10,
                             include_remodelling = True):
    """Perform variational optical flow on the movie.
    This method is experimental and has not been validated.
    
    Parameters:
    -----------
    
    movie : nd array, 3D
        The movie we want to analyse. needs to be a single-colour movie.
        The first dimension is the frame index, the other two dimensions are
        x-pixel index and y-pixel index, respectively.
        
    delta_x : float
        The size of one pixel. This is assumed to be the same in x and y direction
        
    delta_t : float
        The length of the time interval between frames. This is assumed to be the same in x and y direction

    alpha : float
        The lagrange multiplier used in the method.
        
    v_x_guess : float
        The initial guess for v_x

    v_y_guess : float
        The initial guess for v_x
        
    remodelling_guess : float
        The initial guess for the net remodelling
        
    iterations : int
        The number of jacobi iterations that we should use.
        
    include_remodelling : bool
        This argument is ignored, and exists to ensure that this method has the same call signature as our own implementation
        
    Returns:
    --------
    
    v_x : nd array
        The x-velocities. The array has the same dimension as the movie argument, unless return_iterations is True
        
    v_y : nd array
        The y-velocities. The array has the same dimension as the movie argumen, unless return_iterations is True.
    
    net_remodelling : nd array
        The net remodelling. The array has the same dimension as the movie argumen, unless return_iterations is True.
        This will be returned as zeros, we won't use it in this implementation
    """
    # blurred_images= all_images(dtype=float)
    I = movie[0,:,:]
    number_of_frames = movie.shape[0]
    number_of_Xpixels= movie.shape[1]
    number_of_Ypixels= movie.shape[2]
    
    all_v_x = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2), dtype = float)#previous 0.0001*0.019550342130987292
    all_v_y = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2), dtype = float)
    all_remodelling = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2),dtype = float)   
    movie_w_borders = np.zeros((number_of_frames,number_of_Xpixels+2,number_of_Ypixels+2),dtype = float)   
    movie_w_borders[:,1:-1,1:-1] = movie   
    apply_constant_boundary_condition(movie_w_borders[0])

    for frame_index in range(1,movie.shape[0]):
        current_frame = movie_w_borders[frame_index]
        apply_constant_boundary_condition(current_frame)
        previous_frame = movie_w_borders[frame_index -1]
        # next_frame = movie[frame_index +1]
        all_v_x[frame_index -1 ,1:-1,1:-1] = initial_v_x*delta_t/delta_x
        all_v_y[frame_index -1 ,1:-1,1:-1] = initial_v_y*delta_t/delta_x
        all_remodelling[frame_index -1 ,1:-1,1:-1] = initial_remodelling

        v_x = all_v_x[frame_index-1,:,:]
        v_y = all_v_y[frame_index-1,:,:]
        remodelling= all_remodelling[frame_index-1,:,:]
        
        v_x_new = np.copy(v_x)
        v_y_new = np.copy(v_y)
        remodelling_new = np.copy(remodelling)
 
        for iteration_index in range(iterations):
            apply_constant_boundary_condition(v_x)
            apply_constant_boundary_condition(v_y)
            apply_constant_boundary_condition(remodelling)
            for pixel_index_x in range(1,number_of_Xpixels+1):
                for pixel_index_y in range(1,number_of_Ypixels +1):
                    dxdVx_ij = (v_x[pixel_index_x+1,pixel_index_y]-v_x[pixel_index_x-1,pixel_index_y])/2#dVx/dx_ij
                    dydVx_ij = (v_x[pixel_index_x,pixel_index_y+1]-v_x[pixel_index_x,pixel_index_y-1])/2#dVx/dy_ij
                    dxydVx_ij = (v_x[pixel_index_x+1,pixel_index_y+1]
                                 -v_x[pixel_index_x+1,pixel_index_y-1]
                                 -v_x[pixel_index_x-1,pixel_index_y+1]
                                 +v_x[pixel_index_x-1,pixel_index_y-1])/4#d^2Vx/dxdy_ij
                    Vx_barx_ij = v_x[pixel_index_x+1,pixel_index_y]+v_x[pixel_index_x-1,pixel_index_y]

                    this_neighbourhood = np.copy(v_x[pixel_index_x-1:pixel_index_x+2,pixel_index_y-1:pixel_index_y+2])
                    if pixel_index_x ==1:
                        this_neighbourhood[0,:] = 0.0
                    elif pixel_index_x == number_of_Xpixels:
                        this_neighbourhood[2,:] = 0.0
                    if pixel_index_y ==1:
                        this_neighbourhood[:,0] = 0.0
                    elif pixel_index_y == number_of_Ypixels:
                        this_neighbourhood[:,2] = 0.0
 
                    Vx_bar_ij = (this_neighbourhood[0,1]
                                 +this_neighbourhood[2,1]
                                 +this_neighbourhood[1,2]
                                 +this_neighbourhood[1,0]
                                 +this_neighbourhood[0,0]
                                 +this_neighbourhood[0,2]
                                 +this_neighbourhood[2,0]
                                 +this_neighbourhood[2,2])

                    dIdx_ij = (previous_frame[pixel_index_x+1,pixel_index_y]
                               -previous_frame[pixel_index_x-1,pixel_index_y])/2#dI/dx_ij  #h=delta_x in equation
                    dIdy_ij = (previous_frame[pixel_index_x,pixel_index_y+1]
                               -previous_frame[pixel_index_x,pixel_index_y-1])/2#dI/dy_ij  #h=delta_x
                    
                    dIdx_t=(current_frame[pixel_index_x+1,pixel_index_y]
                            -current_frame[pixel_index_x-1,pixel_index_y]
                            -previous_frame[pixel_index_x+1,pixel_index_y]
                            +previous_frame[pixel_index_x-1,pixel_index_y])/2
                    dIdy_t=(current_frame[pixel_index_x,pixel_index_y+1]
                            -current_frame[pixel_index_x,pixel_index_y-1]
                            -previous_frame[pixel_index_x,pixel_index_y+1]
                            +previous_frame[pixel_index_x,pixel_index_y-1])/2

                    #easy form, can compare with the futher improve one
                    dIdt = (current_frame[pixel_index_x,pixel_index_y]
                            -previous_frame[pixel_index_x,pixel_index_y])#It=delta t

                    dIdxx_ij =(previous_frame[pixel_index_x+1,pixel_index_y]
                               +previous_frame[pixel_index_x-1,pixel_index_y]
                               -2*previous_frame[pixel_index_x,pixel_index_y])#Ixx
                    dIdyy_ij =(previous_frame[pixel_index_x,pixel_index_y+1]
                               +previous_frame[pixel_index_x,pixel_index_y-1]
                               -2*previous_frame[pixel_index_x,pixel_index_y])#Iyy
                    dIdyx_ij =(previous_frame[pixel_index_x+1,pixel_index_y+1]
                               -previous_frame[pixel_index_x+1,pixel_index_y-1]
                               -previous_frame[pixel_index_x-1,pixel_index_y+1]
                               +previous_frame[pixel_index_x-1,pixel_index_y-1])/4#Iyx
                    dIdxy_ij = dIdyx_ij #Ixy=Iyx
                    
                    #Evy stencil definitions
                    dxdVy_ij = (v_y[pixel_index_x+1,pixel_index_y]-v_y[pixel_index_x-1,pixel_index_y])/2#dVy/dx_ij
                    dydVy_ij = (v_y[pixel_index_x,pixel_index_y+1]-v_y[pixel_index_x,pixel_index_y-1])/2#dVy/dy_ij
                    dxydVy_ij = (v_y[pixel_index_x+1,pixel_index_y+1]
                                 -v_y[pixel_index_x+1,pixel_index_y-1]
                                 -v_y[pixel_index_x-1,pixel_index_y+1]
                                 +v_y[pixel_index_x-1,pixel_index_y-1])/4#d^2Vy/dxdy_ij
                    Vy_bary_ij = v_y[pixel_index_x,pixel_index_y+1]+v_y[pixel_index_x,pixel_index_y-1]

                    this_neighbourhood = np.copy(v_y[pixel_index_x-1:pixel_index_x+2,pixel_index_y-1:pixel_index_y+2])
                    if pixel_index_x ==1:
                        this_neighbourhood[0,:] = 0.0
                    elif pixel_index_x == number_of_Xpixels:
                        this_neighbourhood[2,:] = 0.0
                    if pixel_index_y ==1:
                        this_neighbourhood[:,0] = 0.0
                    elif pixel_index_y == number_of_Ypixels:
                        this_neighbourhood[:,2] = 0.0
 
                    Vy_bar_ij = (this_neighbourhood[0,1]
                                 +this_neighbourhood[2,1]
                                 +this_neighbourhood[1,2]
                                 +this_neighbourhood[1,0]
                                 +this_neighbourhood[0,0]
                                 +this_neighbourhood[0,2]
                                 +this_neighbourhood[2,0]
                                 +this_neighbourhood[2,2])

                    dyydVy = (v_y[pixel_index_x,pixel_index_y+1]
                              +v_y[pixel_index_x,pixel_index_y-1]
                              -2*v_y[pixel_index_x,pixel_index_y])
                    
                    #Egamma stencil definition
                    remodelling_bar_ij=(remodelling[pixel_index_x-1,pixel_index_y]
                                        +remodelling[pixel_index_x+1,pixel_index_y]
                                        +remodelling[pixel_index_x,pixel_index_y-1]
                                        +remodelling[pixel_index_x,pixel_index_y+1])             
                    remodelling_x=(remodelling[pixel_index_x+1,pixel_index_y]-remodelling[pixel_index_x-1,pixel_index_y])/2 #same as define Vxx, Ix...
                    remodelling_y=(remodelling[pixel_index_x,pixel_index_y+1]-remodelling[pixel_index_x,pixel_index_y-1])/2 
                    
                    #RHS without remodelling
                    F=np.array([-previous_frame[pixel_index_x,pixel_index_y]*(dIdx_t)
                                -previous_frame[pixel_index_x,pixel_index_y]*
                                     (2*dIdx_ij*dxdVx_ij+dIdy_ij*dxdVy_ij+dIdx_ij*dydVy_ij)
                                -previous_frame[pixel_index_x,pixel_index_y]**2*
                                    (Vx_barx_ij+dxydVy_ij)-alpha*Vx_bar_ij,
                                #
                                -previous_frame[pixel_index_x,pixel_index_y]*(dIdy_t)
                                 -previous_frame[pixel_index_x,pixel_index_y]*
                                     (2*dIdy_ij*dydVy_ij+dIdx_ij*dydVx_ij+dIdy_ij*dxdVx_ij)
                                 -previous_frame[pixel_index_x,pixel_index_y]**2*(Vy_bary_ij+dxydVx_ij)-alpha*Vy_bar_ij])
                        
                    # Matrix without remodelling
                    boundary_prefactor = 8
                    if pixel_index_x ==1 or pixel_index_x == number_of_Xpixels:
                        if pixel_index_y == 1 or pixel_index_y == number_of_Ypixels:
                            boundary_prefactor = 3
                        else:
                            boundary_prefactor = 5
                    if pixel_index_y ==1 or pixel_index_y == number_of_Ypixels:
                        if pixel_index_x == 1 or pixel_index_x == number_of_Xpixels:
                            boundary_prefactor = 3
                        else:
                            boundary_prefactor = 5
                            
                    matrix_A=np.array([[previous_frame[pixel_index_x,pixel_index_y]*dIdxx_ij 
                                           -2*previous_frame[pixel_index_x,pixel_index_y]**2
                                            -boundary_prefactor*alpha,
                                        previous_frame[pixel_index_x,pixel_index_y]*dIdyx_ij],
                                       #
                                      [previous_frame[pixel_index_x,pixel_index_y]*dIdxy_ij,
                                       previous_frame[pixel_index_x,pixel_index_y]*dIdyy_ij 
                                           -2*previous_frame[pixel_index_x,pixel_index_y]**2
                                           -boundary_prefactor*alpha]])
                                           #
        
                    matrix_A_inverse=np.linalg.inv(matrix_A)
                    
                    new_state= matrix_A_inverse.dot(F)
                    
                    v_x_new[pixel_index_x,pixel_index_y]=new_state[0]
                    v_y_new[pixel_index_x,pixel_index_y]=new_state[1]

            v_x[:] = v_x_new[:]
            v_y[:] = v_y_new[:]
   
    all_v_x = all_v_x[:,1:-1,1:-1]
    all_v_y = all_v_y[:,1:-1,1:-1]
    all_remodelling = all_remodelling[:,1:-1,1:-1]

    all_v_x *= delta_x/delta_t
    all_v_y *= delta_x/delta_t
    all_speed = np.sqrt(all_v_x**2+all_v_y**2)
    return all_v_x, all_v_y, all_speed, all_remodelling


@njit
def apply_numerical_derivative(matrix, rule):
    """ Apply a numerical derivative to a matrix. assumes that the 
    most outer rows and columsn are dummy entries that should not be returned.
    
    Parameters:
    -----------
    
    matrix : nd array
        The matrix we want to take the numerical derivative of
        
    rule : string
        The derivative we want to take. Can be 'dx', 'dxx', 'dxy', 'dyy'
    
    Returns:
    --------

    derivative : nd array
        The matrix containing the numerical derivative
    """
    if rule == 'dx':
        derivative = (matrix[2:,1:-1] - matrix[:-2, 1:-1])/2
    elif rule == 'dy':
        derivative = (matrix[2:,1:-1] - matrix[:-2, 1:-1])/2
    elif rule == 'dxy' or rule == 'dyx':
        derivative = (matrix[2:,2:] -matrix[2:,:-2] -matrix[:-2,2:] +matrix[:-2,:-2])/4
    elif rule == 'dxx':
        derivative = (matrix[2:,1:-1] +matrix[:-2,1:-1] -2*matrix[1:-1,1:-1])
    elif rule == 'dyy':
        derivative = (matrix[1:-1,2:] +matrix[1:-1,:-2] -2*matrix[1:-1,1:-1])
    elif rule == 'bar_x':
        derivative = matrix[2:,1:-1]+matrix[:-2,1:-1]
    elif rule == 'bar_y':
        derivative = matrix[1:-1,2:]+matrix[1:-1,:-2]
    elif rule == 'bar':
        derivative = ( matrix[:-2,1:-1] +matrix[2:,1:-1] +matrix[1:-1,2:] + matrix[1:-1,:-2] )

    return derivative 

@njit
def variational_optical_flow_jit(movie,
                             delta_x = 1.0,
                             delta_t = 1.0,
                             speed_alpha = 1.0,
                             remodelling_alpha = 1000.0,
                             initial_v_x=np.zeros((10,10),dtype =float),
                             initial_v_y= np.zeros((10,10),dtype =float),
                             initial_remodelling=np.zeros((10,10),dtype =float),
                             max_iterations = 10,
                             tolerance = 1e-10,
                             include_remodelling = True):
    """Perform variational optical flow on the movie.
    This method is experimental and has not been validated.
    
    Parameters:
    -----------
    
    movie : nd array, 3D
        The movie we want to analyse. needs to be a single-colour movie.
        The first dimension is the frame index, the other two dimensions are
        x-pixel index and y-pixel index, respectively.
        
    delta_x : float
        The size of one pixel. This is assumed to be the same in x and y direction
        
    delta_t : float
        The length of the time interval between frames. This is assumed to be the same in x and y direction

    alpha : float
        The lagrange multiplier used in the method.
        
    v_x_guess : float
        The initial guess for v_x

    v_y_guess : float
        The initial guess for v_x
        
    remodelling_guess : float
        The initial guess for the net remodelling
        
    iterations : int
        The number of jacobi iterations that we should use.
        
    tolerance : float
        if the relative difference between successive iterations for all quantities (i.e. v_x, v_y, and remodelling) are less 
        than this tolerance, then method stops

    include_remodelling : bool
        If False, the Liu-shen method will be used, and zeros will be returned for remodelling terms
        
    Returns:
    --------
    
    v_x : nd array
        The x-velocities. The array has the same dimension as the movie argument, unless return_iterations is True
        
    v_y : nd array
        The y-velocities. The array has the same dimension as the movie argumen, unless return_iterations is True.
    
    net_remodelling : nd array
        The net remodelling. The array has the same dimension as the movie argumen, unless return_iterations is True.
    """
    # blurred_images= all_images(dtype=float)
    I = movie[0,:,:]
    number_of_frames = movie.shape[0]
    number_of_Xpixels= movie.shape[1]
    number_of_Ypixels= movie.shape[2]
    
    all_v_x = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2), dtype = float)#previous 0.0001*0.019550342130987292
    all_v_y = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2), dtype = float)
    all_remodelling = np.zeros((number_of_frames-1,number_of_Xpixels+2,number_of_Ypixels+2),dtype = float)   
    movie_w_borders = np.zeros((number_of_frames,number_of_Xpixels+2,number_of_Ypixels+2),dtype = float)   
    movie_w_borders[:,1:-1,1:-1] = movie   
    apply_constant_boundary_condition(movie_w_borders[0])

    for frame_index in range(1,movie.shape[0]):
        previous_frame_w_border = movie_w_borders[frame_index -1]
        current_frame_w_border = movie_w_borders[frame_index]
        apply_constant_boundary_condition(current_frame_w_border)

        previous_frame = previous_frame_w_border[1:-1,1:-1]
        current_frame = current_frame_w_border[1:-1,1:-1]
        # next_frame = movie[frame_index +1]
        all_v_x[frame_index -1 ,1:-1,1:-1] = initial_v_x*delta_t/delta_x
        all_v_y[frame_index -1 ,1:-1,1:-1] = initial_v_y*delta_t/delta_x
        all_remodelling[frame_index -1 ,1:-1,1:-1] = initial_remodelling

        v_x = all_v_x[frame_index-1,:,:]
        v_y = all_v_y[frame_index-1,:,:]
        remodelling= all_remodelling[frame_index-1,:,:]
        
        v_x_new = np.copy(v_x[1:-1,1:-1])
        v_y_new = np.copy(v_y[1:-1,1:-1])
        remodelling_new = np.copy(remodelling[1:-1,1:-1])

        dIdx = apply_numerical_derivative(previous_frame_w_border,'dx')#dI/dx_ij  #h=delta_x in equation
        dIdy = apply_numerical_derivative(previous_frame_w_border,'dy')#dI/dx_ij  #h=delta_x in equation
                
        dIdx_t=(current_frame_w_border[2:,1:-1] -current_frame_w_border[:-2,1:-1]
                    -previous_frame_w_border[2:,1:-1] +previous_frame_w_border[:-2,1:-1])/2

        dIdy_t=(current_frame_w_border[1:-1,2:] -current_frame_w_border[1:-1,:-2]
                    -previous_frame_w_border[1:-1,2:] +previous_frame_w_border[1:-1,:-2])/2

        dIdt = (current_frame_w_border[1:-1,1:-1]
                    -previous_frame_w_border[1:-1,1:-1])#It=delta t

        dIdxx = apply_numerical_derivative(previous_frame_w_border, 'dxx')
        dIdyy = apply_numerical_derivative(previous_frame_w_border, 'dyy')
        dIdyx = apply_numerical_derivative(previous_frame_w_border, 'dyx')

        total_num_iterations = 0
        # last maximum stepsize among v_x, v_y, and remodelling
        last_max_stepsize = 10*tolerance
        
        # This is useful to define outside the loop in case use_remodelling is False
        remodelling_step = 10*tolerance
    
        while total_num_iterations < max_iterations and last_max_stepsize> tolerance:
            apply_constant_boundary_condition(v_x)
            apply_constant_boundary_condition(v_y)
            apply_constant_boundary_condition(remodelling)

            dxdVx_ij = apply_numerical_derivative(v_x,'dx')
            dydVx_ij = apply_numerical_derivative(v_x,'dy')
            dxydVx_ij = apply_numerical_derivative(v_x,'dxy')
            Vx_barx_ij = apply_numerical_derivative(v_x,'bar_x') 
            Vx_bar_ij = apply_numerical_derivative(v_x,'bar') 
            
            dxdVy_ij = apply_numerical_derivative(v_y,'dx')
            dydVy_ij = apply_numerical_derivative(v_y,'dy')
            dxydVy_ij = apply_numerical_derivative(v_y,'dxy')
            Vy_bary_ij = apply_numerical_derivative(v_y,'bar_y') 
            Vy_bar_ij = apply_numerical_derivative(v_y,'bar') 
#
            remodelling_bar = apply_numerical_derivative(remodelling, 'bar')
            remodelling_x = apply_numerical_derivative(remodelling, 'dx')
            remodelling_y = apply_numerical_derivative(remodelling, 'dy')

            if include_remodelling:
                ## RHS
                RHS_x=(previous_frame*(remodelling_x-dIdx_t) -previous_frame*
                                        (2*dIdx*dxdVx_ij+dIdy*dxdVy_ij+dIdx*dydVy_ij)
                                    -previous_frame**2*
                                        (Vx_barx_ij+dxydVy_ij)-speed_alpha*Vx_bar_ij)
                                    #
                RHS_y= (previous_frame*(remodelling_y-dIdy_t) -previous_frame*
                              (2*dIdy*dydVy_ij+dIdx*dydVx_ij+dIdy*dxdVx_ij)
                         -previous_frame**2*(Vy_bary_ij+dxydVx_ij)-speed_alpha*Vy_bar_ij)
                                    #
                RHS_remodelling = +(dIdt+previous_frame*dxdVx_ij
                                     +previous_frame*dydVy_ij
                                     +remodelling_alpha*remodelling_bar)
     
                # LHS matrix
                A11 = (previous_frame*dIdxx -2*previous_frame**2 -4*speed_alpha)
                A12 = previous_frame*dIdyx
                A22 = previous_frame*dIdyy -2*previous_frame**2-4*speed_alpha
                A31 = -dIdx
                A32 = -dIdy
                A33 = +(1+4*remodelling_alpha)
                
                # This is not the actual determinant, but the 2D top left sub-determinant
                det_A = A11*A22 - A12*A12

                # inverse of the matrix
                inv_A11 = A22/det_A
                inv_A12 = -A12/det_A
                inv_A22 = A11/det_A
                inv_A31 = (A32*A12 - A22*A31)/(det_A*A33)
                inv_A32 = (A12*A31 - A32*A11)/(det_A*A33)
                inv_A33 = 1/A33
                
                v_x_new[:] = inv_A11*RHS_x + inv_A12*RHS_y
                v_y_new[:] = inv_A12*RHS_x + inv_A22*RHS_y
                remodelling_new[:] = inv_A31*RHS_x + inv_A32*RHS_y + inv_A33*RHS_remodelling

            else:
                RHS_x = (-previous_frame*dIdx_t -previous_frame*
                                    (2*dIdx*dxdVx_ij+dIdy*dxdVy_ij+dIdx*dydVy_ij)
                            -previous_frame**2*(Vx_barx_ij+dxydVy_ij)
                            -speed_alpha*Vx_bar_ij)
                                #
                RHS_y = (-previous_frame*(dIdy_t) -previous_frame*
                                    (2*dIdy*dydVy_ij+dIdx*dydVx_ij+dIdy*dxdVx_ij)
                            -previous_frame**2*(Vy_bary_ij+dxydVx_ij)
                            -speed_alpha*Vy_bar_ij)
                        
                # Matrix without remodelling
                boundary_prefactor = 4
                A_11 = previous_frame*dIdxx -2*previous_frame**2 -boundary_prefactor*speed_alpha 
                A_12 = previous_frame*dIdyx
                A_22 = previous_frame*dIdyy -2*previous_frame**2 -boundary_prefactor*speed_alpha
                                           
                det_A = A_11*A_22-A_12*A_12
                
                inv_A_11 = 1/det_A*A_22
                inv_A_12 = -1/det_A*A_12
                inv_A_22 = 1/det_A*A_11
        
                v_x_new[:] = inv_A_11*RHS_x + inv_A_12*RHS_y
                v_y_new[:] = inv_A_12*RHS_x + inv_A_22*RHS_y
                    
            # relaxation_factor = 1.1
            # v_x_new = (1-relaxation_factor)*v_x[1:-1,1:-1] + relaxation_factor*v_x_new
            # v_y_new = (1-relaxation_factor)*v_y[1:-1,1:-1] + relaxation_factor*v_y_new
            v_x_step = np.linalg.norm(v_x[1:-1,1:-1] - v_x_new)/ np.linalg.norm(v_x_new)
            v_y_step = np.linalg.norm(v_y[1:-1,1:-1] - v_y_new)/ np.linalg.norm(v_y_new)
            v_x[1:-1,1:-1] = v_x_new
            v_y[1:-1,1:-1] = v_y_new
            if include_remodelling:
                # remodelling_new = (1-relaxation_factor)*remodelling[1:-1,1:-1] + relaxation_factor*remodelling_new
                remodelling_step = np.linalg.norm(remodelling[1:-1,1:-1] - remodelling_new)/ np.linalg.norm(remodelling_new)
                remodelling[1:-1,1:-1] = remodelling_new
                
            last_max_stepsize = max((v_x_step, v_y_step, remodelling_step))

            total_num_iterations +=1
            if total_num_iterations%10000 == 0:
                print('passed ' + str(total_num_iterations) + ' iterations, the current stepsize in v_x, v_y, and remodelling are: ' )
                print(v_x_step) 
                print(v_y_step) 
                print(remodelling_step) 
                print('')
   
    all_v_x = all_v_x[:,1:-1,1:-1]
    all_v_y = all_v_y[:,1:-1,1:-1]
    all_remodelling = all_remodelling[:,1:-1,1:-1]

    all_v_x *= delta_x/delta_t
    all_v_y *= delta_x/delta_t
    all_speed = np.sqrt(all_v_x**2+all_v_y**2)
    return all_v_x, all_v_y, all_speed, all_remodelling, total_num_iterations

@njit
def apply_constant_boundary_condition(image = np.zeros((10,10),dtype = float)):
    """apply periodic boundary conditions on an image. The image is edited in-place
    
    Parameters:
    -----------
    
    image : 2D array
    """
    image[0,:] = image[1,:]
    image[-1,:] = image[-2,:]
    image[:,0] = image[:,1]
    image[:,-1] = image[:,-2]

def conduct_variational_optical_flow(movie, 
                                     delta_x = 1.0,
                                     delta_t = 1.0,
                                     speed_alpha = 1.0,
                                     remodelling_alpha = 1000.0,
                                     v_x_guess=0.1,
                                     v_y_guess= 0.1,
                                     remodelling_guess=0.5,
                                     max_iterations = 10,
                                     smoothing_sigma = None,
                                     return_iterations = False,
                                     iteration_stepsize = 1,
                                     tolerance = 1e-10,
                                     include_remodelling = True,
                                     use_liu_shen = False):
    """Conduct optical flow as in Vig et al. Biophysical Journal 110, 1469–1475, 2016.
    
    Parameters:
    -----------

    movie : np.array
        the movie we wish to analyse
    
    boxsize : int
        the boxsize for the optical flow algorithm. If an even number is provided, the next smallest uneven number will be used.

    delta_x : float
        the size of one pixel in the movie. We assume pixel size is identical in the x and y directions

    delta_t : float
        the time interval between frames in the movie. Defaults to 1.0.
        
    alpha : float
        The lagrange multiplier used in the method.
        
    v_x_guess : float
        The initial guess for v_x

    v_y_guess : float
        The initial guess for v_x
        
    remodelling_guess : float
        The initial guess for the net remodelling
        
    max_iterations : int
        The number of jacobi iterations that we should use.
 
    smoothing_sigma : float or None
        If the value None is provided, no smoothing will be applied. Otherwise a gaussian blur with this sigma value
        will be applied to the movie before optical flow is conducted.
        
    return_iterations : bool
        if True then return the output from each iteration, rather than just the last iteration
        
    iteration_stepsize : int
        Only relevant if return_iterations is true, this argument will be ignored otherwise
        if return_iterations is True, this will enforce that iterations are subsampled with this interval,
        i.e. only ever iteration_stepsize step will be recorded
    
    tolerance : float
        if the relative difference between successive iterations for all quantities (i.e. v_x, v_y, and remodelling) are less 
        than this tolerance, then method stops
    
    include_remodelling : bool
        whether to include remodelling in the calculation. If this is False, the Liu-Shen method will be used
    
    use_liu_shen : bool
        if True, then an exact re-implementation of Liu-Shen's algorithm will be used.

    Returns:
    --------

    result : dict
        A dictionary containing the results of optical flow calculations, as well as arrays for the orignal and blurred data.
        The keys are: v_x, v_y, speed, original_data, blurred_data, delta_x, delta_t
    """
    if smoothing_sigma is not None:
        movie_to_analyse = blur_movie(movie, smoothing_sigma=smoothing_sigma)
    else:
        movie_to_analyse = movie
        
    if use_liu_shen:
        optical_flow_method = liu_shen_optical_flow_jit
    else: 
        optical_flow_method = variational_optical_flow_jit

    initial_v_x = np.full((movie.shape[1],movie.shape[2]),float(v_x_guess)) 
    initial_v_y = np.full((movie.shape[1],movie.shape[2]),float(v_y_guess)) 
    initial_remodelling = np.full((movie.shape[1],movie.shape[2]),float(remodelling_guess)) 
    print('starting optical flow calculations')
    print('')
    print('initial matrices have the following means')
    print('v_x:')
    print(np.mean(initial_v_x))
    print('v_y:')
    print(np.mean(initial_v_y))
    print('remodelling:')
    print(np.mean(initial_remodelling))
    print('speed:')
    print(np.mean(np.sqrt(initial_v_x**2 + initial_v_y**2)))

    result = dict()
    if return_iterations:
        this_v_x, this_v_y, this_speed, this_remodelling, this_iteration_number = optical_flow_method(movie_to_analyse, 
                                                                                    delta_x,
                                                                                    delta_t,
                                                                                    speed_alpha,
                                                                                    remodelling_alpha,
                                                                                    initial_v_x,
                                                                                    initial_v_y,
                                                                                    initial_remodelling,
                                                                                    max_iterations = iteration_stepsize,
                                                                                    tolerance = tolerance,
                                                                                    include_remodelling = include_remodelling)

        print('at iteration ' + str(iteration_stepsize))
        print('this mean_speed is')
        print(np.mean(this_speed))
        print('this mean remodelling iis')
        print(np.mean(this_remodelling))
        number_of_recorded_iterations = max_iterations//iteration_stepsize

        all_v_x = np.zeros((this_v_x.shape[0], number_of_recorded_iterations + 1, this_v_x.shape[1], this_v_x.shape[2])) 
        all_v_y = np.zeros_like(all_v_x)
        all_speed = np.zeros_like(all_v_x)
        all_remodelling = np.zeros_like(all_v_x)

        all_v_x[:,0,:,:] = initial_v_x
        all_v_y[:,0,:,:] = initial_v_x
        all_speed[:,0,:,:] = np.sqrt(initial_v_x**2 + initial_v_y**2)
        all_remodelling[:,0,:,:] = initial_remodelling


        all_v_x[:,1,:,:] = this_v_x
        all_v_y[:,1,:,:] = this_v_y
        all_speed[:,1,:,:] = this_speed
        all_remodelling[:,1,:,:] = this_remodelling

        tolerance_reached = False
        total_iterations = max_iterations
        for iteration_index in range(2,number_of_recorded_iterations+1):
            this_v_x, this_v_y, this_speed, this_remodelling, this_iteration_number = optical_flow_method(movie_to_analyse, 
                                                                                    delta_x,
                                                                                    delta_t,
                                                                                    speed_alpha,
                                                                                    remodelling_alpha,
                                                                                    this_v_x,
                                                                                    this_v_y,
                                                                                    this_remodelling,
                                                                                    max_iterations = iteration_stepsize,
                                                                                    tolerance = tolerance,
                                                                                    include_remodelling = include_remodelling)
            print('at iteration ' + str(iteration_index*iteration_stepsize))
            print('this mean_speed is')
            print(np.mean(this_speed))
            print('this mean remodelling is')
            print(np.mean(this_remodelling))
            print('')

            all_v_x[:,iteration_index,:,:] = this_v_x
            all_v_y[:,iteration_index,:,:] = this_v_y
            all_speed[:,iteration_index,:,:] = this_speed
            all_remodelling[:,iteration_index,:,:] = this_remodelling
            
            if this_iteration_number< iteration_stepsize and not tolerance_reached:
                total_iterations = (iteration_index -1) * iteration_stepsize + this_iteration_number
                print('')
                print('I reached the required tolerance within ' + str(total_iterations) + ' iterations. All remaining iteration steps will only contain one step instead of ' + str(iteration_index))
                print('')
                print(tolerance_reached)

        result['v_x'] = all_v_x[:,-1,:,:]
        result['v_y'] = all_v_y[:,-1,:,:]
        result['speed'] = all_speed[:,-1,:,:]
        result['remodelling'] = all_remodelling[:,-1,:,:]

        result['v_x_steps'] = all_v_x
        result['v_y_steps'] = all_v_y
        result['speed_steps'] = all_speed
        result['remodelling_steps'] = all_remodelling
        result['iteration_stepsize'] = iteration_stepsize

    else:
        all_v_x, all_v_y, all_speed, all_remodelling, total_iterations = optical_flow_method(movie_to_analyse, 
                                                                                    delta_x,
                                                                                    delta_t,
                                                                                    speed_alpha,
                                                                                    remodelling_alpha,
                                                                                    initial_v_x,
                                                                                    initial_v_y,
                                                                                    initial_remodelling,
                                                                                    max_iterations,
                                                                                    tolerance = tolerance,
                                                                                    include_remodelling = include_remodelling)
        result['v_x'] = all_v_x
        result['v_y'] = all_v_y
        result['speed'] = all_speed
        result['remodelling'] = all_remodelling

    print('')
    print('optical flow calculations finished after ' + str(total_iterations) + ' iterations')
    print('')
    result['original_data'] = movie
    result['delta_x'] = delta_x
    result['delta_t'] = delta_t
    result['blurred_data'] = all_remodelling
    result['max_iterations'] = max_iterations
    result['total_iterations'] = total_iterations
    
    return result

def costum_imshow(image, delta_x, cmap = 'gray_r', autoscale = False, v_min = 0.0, v_max = 255.0):
    """Our typical way to show images. Will display the image without any anti-aliasing and add axis units and labels. 
    Can be used for simulated images if autoscale is set to True. The figure and figure panels need to be created outside
    of this function.
    
    Parameters :
    ------------
    
    image : np array (2D)
        The image we wish to display
        
    delta_x : float
        the size of one pixel
        
    cmap : string
        name of a matplotlib color map
        
    autoscale : bool
        if True, the image will be displayed using matplotlib's autoscaling.
        Otherwise, the image will be displayed using the scale (v_min,v_max)
        
    v_min : float
        pixels equal to or below this image intensity will be white in inverted grayscale
        
    v_max : float
        pixels equal to or above this image intensity will be black in inverted grayscale
    
    """
    if autoscale:
        v_min = None
        v_max = None
    else: 
        v_min = v_min
        v_max = v_max

    Xpixels=image.shape[0]#Number of X pixels=379
    Ypixels=image.shape[1]#Number of Y pixels=279
    x_extent = Xpixels * delta_x
    y_extent = Ypixels * delta_x
    plt.imshow(image,cmap = cmap, extent = [0,y_extent, x_extent, 0], vmin = v_min, vmax = v_max, interpolation = None)
    plt.xlabel("y-position [$\mathrm{\mu}$m]")
    plt.ylabel("x-position [$\mathrm{\mu}$m]")
 
def subsample_velocities_for_visualisation(flow_result, iteration = None, arrow_boxsize = 5):
    """Generate arrows for plotting from a flow result. Will generate quantities that
    can be passed to plt.quiver.
    
    Parameters :
    ------------
    
    flow_result : dict
        output of our optical flow calculations
    
    iteration : int or none
        if provided, the velocites are read out at this iteation, rather than the final, converged result

    arrow_boxsize : int
        size of the box around each arrow in pixels
        
    Returns :
    ---------
    x_positions : np.array
        vector of x positions for plotting the arrows at

    y_positions : np.array
        vector of y positions for plotting the arrows at

    v_x : np.array
        vector of x-velocities

    v_y : np.array
        vector of y-velocities 
    """
    movie = flow_result['original_data']

    number_of_frames = movie.shape[0]
    Xpixels=flow_result['v_x'].shape[1]#Number of X pixels=379
    Ypixels=flow_result['v_y'].shape[2]#Number of Y pixels=279
    x_extent = Xpixels * flow_result['delta_x']
    y_extent = Ypixels * flow_result['delta_x']
    Nbx = int(Xpixels/arrow_boxsize)
    Nby = int(Ypixels/arrow_boxsize)
 
    subsampled_v_x  = np.zeros((number_of_frames-1,Nbx,Nby))
    subsampled_v_y = np.zeros((number_of_frames-1,Nbx,Nby))
    for frame_index in range(1,movie.shape[0]):
        this_subsampled_v_x = subsampled_v_x[frame_index-1,:,:]
        this_subsampled_v_y = subsampled_v_y[frame_index-1,:,:]        
        for arrow_box_index_x in range(Nbx):
            for arrow_box_index_y in range(Nby):
                if iteration is not None:
                    v_x_at_pixel = flow_result['v_x_steps'][frame_index-1,iteration,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
                    v_y_at_pixel = flow_result['v_y_steps'][frame_index-1,iteration,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
                else:
                    v_x_at_pixel = flow_result['v_x'][frame_index-1,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
                    v_y_at_pixel = flow_result['v_y'][frame_index-1,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
                this_subsampled_v_x[arrow_box_index_x,arrow_box_index_y] = v_x_at_pixel 
                this_subsampled_v_y[arrow_box_index_x,arrow_box_index_y] = v_y_at_pixel 
                
    upper_mgrid_limit_x = int(Nbx*arrow_boxsize)
    upper_mgrid_limit_y = int(Nby*arrow_boxsize)#to make1024/100=100
    x_positions = np.mgrid[0:upper_mgrid_limit_x:arrow_boxsize]
    x_positions += round(arrow_boxsize/2)
    x_positions = x_positions.astype('float')
    y_positions = np.mgrid[0:upper_mgrid_limit_y:arrow_boxsize]
    y_positions += round(arrow_boxsize/2)
    y_positions = y_positions.astype('float')
    
    x_positions = x_positions/Xpixels*x_extent
    y_positions = y_positions/Ypixels*y_extent
    
    return x_positions, y_positions, subsampled_v_x, subsampled_v_y

 
def make_velocity_overlay_movie(flow_result,filename, arrow_boxsize = 5, arrow_scale = 1.0, cmap = 'gray_r', 
                                autoscale = False, arrow_color = 'magenta', v_min = 0.0, v_max = 255.0):   
    """Plot a optical flow velocity result
    
    Parameters :
    ------------
    
    flow_result : dict
        output of our optical flow calculations
    
    filename : string
        saving location
        
    boxsize : int
        size of the box around each arrow in pixels
        
    arrow_scale : float
        scaling paramter to change the length of arrows
        
    cmap : string
        name of a matplotlib colormap to be used
        
    arrow_color : string
        maptlotlib name of the color of the arrows
        
    autoscale : bool
        if True, the image will be displayed using matplotlib's autoscaling.
        Otherwise, the image will be displayed using the scale (v_min,v_max)
        
    v_min : float
        pixels equal to or below this image intensity will be white in inverted grayscale
        
    v_max : float
        pixels equal to or above this image intensity will be black in inverted grayscale
    """
    movie = flow_result['original_data']
   
    x_positions, y_positions, v_x, v_y = subsample_velocities_for_visualisation(flow_result, arrow_boxsize = arrow_boxsize)

    fig = plt.figure(figsize = (2.5,2.5))
    def animate(i): 
        plt.cla()
        costum_imshow(movie[i+1,:,:], delta_x = flow_result['delta_x'], cmap = cmap, autoscale=autoscale, v_min = v_min, v_max = v_max)
        plt.quiver(y_positions, x_positions, v_y[i,:,:], -v_x[i,:,:], color = arrow_color,headwidth=5, scale = 1.0/arrow_scale)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=movie.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')
    ani.save(filename,dpi=600) 

def make_convergence_plots(result, filename_start):
    """ Make a compound convergence plot for variational optical flow.
    
    Multiple figures will be saved with filenames that start with filename_start and have varying extensions
    
    Only the frist pair of frames will be plotted for the convergence analysis.
    
    Parameters:
    -----------
    
    result : dict
        the output of the variational optical flow method. The opitical flow method needs to have been called with 
        return_iterations = True
        
    filename_starts : string
        The start of the filename
        
    Returns nothing.
    """
    iterations = result['max_iterations']
    iteration_stepsize = result['iteration_stepsize']
    delta_x = result['delta_x']
    original_data = result['original_data']

    speed_error = np.zeros(iterations//iteration_stepsize)
    remodelling_error = np.zeros(iterations//iteration_stepsize)
    stepsizes = np.arange(0,iterations+0.5,iteration_stepsize,dtype = int)
    for iteration_index in range(iterations//iteration_stepsize):
        this_speed_error = (np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:] - 
                                             result['speed_steps'][0,iteration_index,:,:])/
                         np.linalg.norm(result['speed_steps'][0,iteration_index + 1,:,:]))
        speed_error[iteration_index] = this_speed_error

        this_remodelling_error = (np.linalg.norm(result['remodelling_steps'][0,iteration_index + 1,:,:] - 
                                             result['remodelling_steps'][0,iteration_index,:,:])/
                         np.linalg.norm(result['remodelling_steps'][0,iteration_index + 1,:,:]))
        remodelling_error[iteration_index] = this_remodelling_error
        
    plt.figure(figsize = (2.5,2.5), constrained_layout = True) 
    plt.plot(stepsizes[1:], speed_error)
    plt.title('Speed stepsize per ' + str(iteration_stepsize) + '\niterations')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('relative step size')
    plt.savefig(filename_start + 'speed_convergence.pdf')
 
    plt.figure(figsize = (2.5,2.5), constrained_layout = True) 
    plt.plot(stepsizes[1:], remodelling_error)
    plt.title('Remodelling stepsize per ' + str(iteration_stepsize) + '\niterations')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('relative step size')
    plt.savefig(filename_start + 'remodelling_convergence.pdf')
 
    # optical_flow.make_velocity_overlay_movie(result, 
                                            #  os.path.join(os.path.dirname(__file__),'output',
                                                        #   'variational_test_iterations_w_noise_' + str(iterations) + '.mp4'), 
                                            #  autoscale = True,
                                            #  arrow_scale = 0.5,
                                            #  arrow_boxsize = 4)
    
    fig = plt.figure(figsize = (6.5,4.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.suptitle('Iteration ' + str(stepsizes[i]))

        x_positions, y_positions, v_x, v_y = subsample_velocities_for_visualisation(result, iteration =i, arrow_boxsize = 4)
        plt.subplot(231)
        costum_imshow(original_data[0,:,:],delta_x = delta_x,v_min = np.min(original_data[0,:,:]), v_max = np.max(original_data[:,0,0]))
        plt.quiver(y_positions, x_positions, v_y[0,:,:], -v_x[0,:,:], color = 'magenta',headwidth=5, scale = None)
        plt.xlabel('')

        plt.subplot(232)
        this_speed_frame = np.zeros((original_data.shape[1], original_data.shape[2]))
        this_speed_frame[:,:] = result['speed_steps'][0,i,:,:]
        costum_imshow(this_speed_frame,delta_x = delta_x, autoscale = True, cmap = 'viridis')
        plt.xlabel('')
        plt.ylabel('')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(result['speed_steps']),np.max(result['speed_steps']))
        # colorbar.formatter = matplotlib.ticker.StrMethodFormatter("{x:." + str(2) + "f}")
        plt.title('Motion speed [$\mathrm{\mu m}$/s]')

        plt.subplot(233)
        this_remodelling_frame = np.zeros((original_data.shape[1], original_data.shape[2]))
        this_remodelling_frame[:,:] = result['remodelling_steps'][0,i,:,:]
        costum_imshow(this_remodelling_frame,delta_x = delta_x, autoscale = True, cmap = 'plasma')
        plt.ylabel('')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(result['remodelling_steps']),np.max(result['remodelling_steps']))
        plt.title('Net remodelling')

        plt.subplot(234)
        this_remodelling_frame = np.zeros((original_data.shape[1], original_data.shape[2]))
        this_remodelling_frame[:,:] = result['v_x_steps'][0,i,:,:]
        costum_imshow(this_remodelling_frame,delta_x = delta_x, autoscale = True, cmap = 'viridis')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(result['v_x_steps']),np.max(result['v_x_steps']))
        plt.title('x velocity [$\mathrm{\mu m}$/s]')

        plt.subplot(235)
        this_remodelling_frame = np.zeros((original_data.shape[1], original_data.shape[2]))
        this_remodelling_frame[:,:] = result['v_y_steps'][0,i,:,:]
        costum_imshow(this_remodelling_frame,delta_x = delta_x, autoscale = True, cmap = 'viridis')
        plt.ylabel('')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(result['v_y_steps']),np.max(result['v_y_steps']))
        plt.title('y velocity [$\mathrm{\mu m}$/s]')

        plt.subplot(236)
        plt.plot(stepsizes[1:], speed_error)
        plt.title('Speed stepsize per ' + str(iteration_stepsize) + '\niterations')
        if i<len(stepsizes) -1:
            plt.scatter(stepsizes[i+1],speed_error[i])
        else:
            plt.scatter(stepsizes[-1],speed_error[-1])
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('relative step size')
 
    ani = FuncAnimation(fig, animate, frames=result['speed_steps'].shape[1])
    ani.save(filename_start + 'compound_figures.mp4',dpi=300)
 
def make_joint_overlay_movie(flow_result,filename, arrow_boxsize = 5, arrow_scale = 1.0, cmap = 'gray_r', 
                                autoscale = False, arrow_color = 'magenta', v_min = 0.0, v_max = 255.0):   
    """Plot a optical flow velocity result. In addition to the raw data and arrows, it also includes the 
    speed and remodelling.
    
    Parameters :
    ------------
    
    flow_result : dict
        output of our optical flow calculations
    
    filename : string
        saving location
        
    boxsize : int
        size of the box around each arrow in pixels
        
    arrow_scale : float
        scaling paramter to change the length of arrows
        
    cmap : string
        name of a matplotlib colormap to be used
        
    arrow_color : string
        maptlotlib name of the color of the arrows
        
    autoscale : bool
        if True, the image will be displayed using matplotlib's autoscaling.
        Otherwise, the image will be displayed using the scale (v_min,v_max)
        
    v_min : float
        pixels equal to or below this image intensity will be white in inverted grayscale
        
    v_max : float
        pixels equal to or above this image intensity will be black in inverted grayscale
    """
    original_data = flow_result['original_data']
   
    x_positions, y_positions, v_x, v_y = subsample_velocities_for_visualisation(flow_result, arrow_boxsize = arrow_boxsize)

    fig = plt.figure(figsize = (6.5,2.5), constrained_layout = True)
    def animate(i): 
        plt.clf()
        plt.subplot(131)
        costum_imshow(original_data[i,:,:], delta_x = flow_result['delta_x'], cmap = cmap, autoscale=autoscale, v_min = v_min, v_max = v_max)
        plt.quiver(y_positions, x_positions, v_y[i,:,:], -v_x[i,:,:], color = arrow_color,headwidth=5, scale = 1.0/arrow_scale)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        
        plt.subplot(132)
        costum_imshow(flow_result['speed'][i,:,:],delta_x = flow_result['delta_x'], autoscale = True, cmap = 'viridis')
        plt.ylabel('')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(flow_result['speed']),np.max(flow_result['speed']))
        # colorbar.formatter = matplotlib.ticker.StrMethodFormatter("{x:." + str(2) + "f}")
        plt.title('Motion speed [$\mathrm{\mu m}$/s]')

        plt.subplot(133)
        costum_imshow(flow_result['remodelling'][i,:,:],delta_x = flow_result['delta_x'], autoscale = True, cmap = 'plasma')
        plt.ylabel('')
        colorbar = plt.colorbar(shrink = 0.6)
        plt.clim(np.min(flow_result['remodelling']),np.max(flow_result['remodelling']))
        # colorbar.formatter = matplotlib.ticker.StrMethodFormatter("{x:." + str(2) + "f}")
        plt.title('Net remodelling')

    ani = FuncAnimation(fig, animate, frames=original_data.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')
    ani.save(filename,dpi=600) 