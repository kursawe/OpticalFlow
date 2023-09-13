import numpy as np
from numba import jit
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
from matplotlib.animation import FuncAnimation

@jit(nopython=True, error_model = "numpy")
def conduct_optical_flow_jit(movie, box_size = 15, delta_x = 1.0):
    
    # movie = np.array(movie, dtype = np.float64)
    movie = movie.astype(np.float64)
    first_frame = movie[0,:,:]
    dIdx = np.zeros_like(first_frame)
    delta_t = 1
    number_of_frames = movie.shape[0]
    #Nb = int((1024-2)/box_size)#Number of boxes
    Xpixels=movie.shape[1]#Number of X pixels=379
    Ypixels=movie.shape[2]#Number of Y pixels=279
    Nbx= int((Xpixels-2)/box_size)
    Nby= int((Ypixels-2)/box_size)
    # all_v_x = np.zeros((number_of_frames-1,Nbx*box_size,Nby*box_size))
    # all_v_y = np.zeros((number_of_frames-1,Nbx*box_size,Nby*box_size))
    all_v_x = np.zeros((number_of_frames-1,Xpixels,Ypixels))
    all_v_y = np.zeros((number_of_frames-1,Xpixels,Ypixels))
    #get average speed of each frame:|v|=np.sqrt(Vx**2+Vy**2), np.mean(V)
    all_speed= np.zeros((number_of_frames-1,Xpixels,Ypixels))
    
    print('conducting optical flow now')
    for frame_index in range(1,movie.shape[0]):
        print('current_frame: ' + str(frame_index))
        current_frame = movie[frame_index]
        previous_frame = movie[frame_index -1]
            
        dIdx = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/(4*delta_t)
        dIdy = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/(4*delta_t)

        delta_I_too_big = current_frame-previous_frame
       
        delta_I = delta_I_too_big[1:Xpixels-1,1:Ypixels-1]#0-1024 in total
        #In other words
        ##delta_I = delta_I_too_big[1:-1,1:-1]
        
        b = box_size*box_size 
        v_x = all_v_x[frame_index-1,:,:]
        v_y = all_v_y[frame_index-1,:,:]
        
        speed= all_speed[frame_index-1,:,:]
        
        for pixel_index_x in range(Nbx*box_size):
            #from0,...Nb-1
            for pixel_index_y in range(Nby*box_size):#better chose odd number box_size to make sure it's symmetrical
                x_begining = (max(pixel_index_x-int(box_size/2),0))#e.g box_size=11,half is 5 and pixel in center
                x_end = (min(pixel_index_x+int(box_size/2)+1,int(Nbx*box_size)))#e.g box_size=11,pixel=20 then it's 26 but stop at 25
                y_beginning = (max(pixel_index_y-int(box_size/2),0))
                y_end = (min(pixel_index_y+int(box_size/2)+1,int(Nby*box_size)))
                
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
        v_x*= delta_x
        v_y*= delta_x
        speed*=delta_x
        
    return all_v_x, all_v_y, all_speed, movie

def conduct_optical_flow(movie, boxsize = 15, delta_x = 1.0):
    all_v_x, all_v_y, all_speed, movie = conduct_optical_flow_jit(movie, boxsize, delta_x)
    result = dict()
    result['v_x'] = all_v_x
    result['v_y'] = all_v_y
    result['speed'] = all_speed
    result['original_data'] = movie
    result['delta_x'] = delta_x

    return result
    
def make_velocity_overlay_movie(flow_result,filename, boxsize = 5, arrow_scale = 1.0):   
    movie = flow_result['original_data']
    number_of_frames = movie.shape[0]
    Xpixels=flow_result['v_x'].shape[1]#Number of X pixels=379
    Ypixels=flow_result['v_y'].shape[2]#Number of Y pixels=279
    Nbx = int(Xpixels/boxsize)
    Nby = int(Ypixels/boxsize)
    
    newall_v_x  = np.zeros((number_of_frames-1,Nbx,Nby))
    newall_v_y = np.zeros((number_of_frames-1,Nbx,Nby))
    for frame_index in range(1,movie.shape[0]):
        newall_v_x_box = newall_v_x[frame_index-1,:,:]
        newall_v_y_box = newall_v_y[frame_index-1,:,:]        
        for arrow_box_index_x in range(Nbx):
            for arrow_box_index_y in range(Nby):
                this_newall_v_x_box = flow_result['v_x'][frame_index-1,arrow_box_index_x*boxsize + round(boxsize/2),arrow_box_index_y*boxsize + round(boxsize/2)]
                newall_v_x_box[arrow_box_index_x,arrow_box_index_y] = this_newall_v_x_box 
                this_newall_v_y_box = flow_result['v_y'][frame_index-1,arrow_box_index_x*boxsize + round(boxsize/2),arrow_box_index_y*boxsize + round(boxsize/2)]
                newall_v_y_box[arrow_box_index_x,arrow_box_index_y] = this_newall_v_y_box 
    fig = plt.figure(figsize = (2.5,2.5))
    def animate(i): 
        plt.cla()
        upper_mgrid_limit_x = int(Nbx*boxsize)
        upper_mgrid_limit_y = int(Nby*boxsize)#to make1024/100=100
        x_pos = np.mgrid[0:upper_mgrid_limit_x:boxsize]
        x_pos += round(boxsize/2)
        y_pos = np.mgrid[0:upper_mgrid_limit_y:boxsize]
        y_pos += round(boxsize/2)
        x_direct = newall_v_x[i,:,:]
        y_direct = newall_v_y[i,:,:]      
        plt.imshow(movie[i,:,:])
        plt.quiver(y_pos, x_pos, y_direct, -x_direct, color = 'white',headwidth=5, scale = 1.0/arrow_scale)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
        plt.title("Visualizing Velocity") 
        plt.xlabel("Number of Pixels")
        plt.ylabel("Number of Pixels")
        plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=movie.shape[0]-1)
    #ani.save('Visualizing Velocity.gif')
    ani.save(filename,dpi=600) 

