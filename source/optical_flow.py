import numpy as np
from numba import jit
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
from matplotlib.animation import FuncAnimation
import skimage.filters

@jit(nopython=True, error_model = "numpy")
def conduct_optical_flow_jit(movie, box_size = 15, delta_x = 1.0, delta_t = 1.0):
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
        
    Returns :
    ---------
    
    v_x : np array
        The calculated x-velocities at each pixel

    v_y : np array
        The calculated y-velocities at each pixel

    speed : np array
        The speed at each pixel
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
                #ABC only
                #if mode == 'velocity_only':
                A = np.sum((local_dIdx)**2)
                B = np.sum(local_dIdx*local_dIdy)
                C = np.sum((local_dIdy)**2)
                Vx =(-C*sum1+ B*sum2)/(delta_t*(A*C-B**2))
                Vy =(-A*sum2+ B*sum1)/(delta_t*(A*C-B**2))
                Vspeed=np.sqrt(Vx**2+Vy**2)

                v_x[pixel_index_x,pixel_index_y] = Vx
                v_y[pixel_index_x,pixel_index_y] = Vy
                
                speed[pixel_index_x,pixel_index_y] = Vspeed
        v_x*= delta_x/delta_t_data
        v_y*= delta_x/delta_t_data
        speed*=delta_x/delta_t_data
        
    return all_v_x, all_v_y, all_speed

def conduct_optical_flow(movie, boxsize = 15, delta_x = 1.0, delta_t = 1.0, smoothing_sigma = None):
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

    all_v_x, all_v_y, all_speed = conduct_optical_flow_jit(movie_to_analyse, boxsize, delta_x, delta_t)
    result = dict()
    result['v_x'] = all_v_x
    result['v_y'] = all_v_y
    result['speed'] = all_speed
    result['original_data'] = movie
    result['delta_x'] = delta_x
    result['delta_t'] = delta_t
    result['blurred_data'] = movie_to_analyse

    return result

def blur_movie(movie, smoothing_sigma):
    """"Blur a movie with the given sigma value.
    
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

   
def costum_imshow(image, delta_x, cmap = 'gray_r', autoscale = False):
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
        Otherwise, the image will be displayed using the scale (0,255)
    
    """
    if autoscale:
        v_min = None
        v_max = None
    else: 
        v_min = 0.0
        v_max = 255.0

    Xpixels=image.shape[0]#Number of X pixels=379
    Ypixels=image.shape[1]#Number of Y pixels=279
    x_extent = Xpixels * delta_x
    y_extent = Ypixels * delta_x
    plt.imshow(image,cmap = cmap, extent = [0,y_extent, x_extent, 0], vmin = v_min, vmax = v_max, interpolation = None)
    plt.xlabel("y-position [$\mathrm{\mu}$m]")
    plt.ylabel("x-position [$\mathrm{\mu}$m]")
 
def subsample_velocities_for_visualisation(flow_result, arrow_boxsize = 5):
    """Generate arrows for plotting from a flow result. Will generate quantities that
    can be passed to plt.quiver.
    
    Parameters :
    ------------
    
    flow_result : dict
        output of our optical flow calculations
    
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
                v_x_at_pixel = flow_result['v_x'][frame_index-1,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
                this_subsampled_v_x[arrow_box_index_x,arrow_box_index_y] = v_x_at_pixel 
                v_y_at_pixel = flow_result['v_y'][frame_index-1,arrow_box_index_x*arrow_boxsize + round(arrow_boxsize/2),arrow_box_index_y*arrow_boxsize + round(arrow_boxsize/2)]
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

 
def make_velocity_overlay_movie(flow_result,filename, arrow_boxsize = 5, arrow_scale = 1.0, cmap = 'gray_r', autoscale = False, arrow_color = 'magenta'):   
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
    """
    movie = flow_result['original_data']
   
    x_positions, y_positions, v_x, v_y = subsample_velocities_for_visualisation(flow_result, arrow_boxsize = arrow_boxsize)

    fig = plt.figure(figsize = (2.5,2.5))
    def animate(i): 
        plt.cla()
        costum_imshow(movie[i,:,:], delta_x = flow_result['delta_x'], cmap = cmap, autoscale=autoscale)
        plt.quiver(y_positions, x_positions, v_y[i,:,:], -v_x[i,:,:], color = arrow_color,headwidth=5, scale = 1.0/arrow_scale)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        if i <1:
            plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=movie.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')
    ani.save(filename,dpi=600) 
