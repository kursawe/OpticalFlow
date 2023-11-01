import muDIC as dic
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io


sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow
import compare_rho_and_actin
delta_x = compare_rho_and_actin.delta_x
delta_t = compare_rho_and_actin.delta_t

from matplotlib.animation import FuncAnimation

def try_mu_dic():

    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    # actin_movie_blurred = optical_flow.blur_movie(actin_movie, smoothing_sigma = 1.3)
    path = os.path.join(os.path.dirname(__file__), 'data', 'actin_image_sequence_blurred')
    # use dic.image_stack_from_list
    # image_list = [actin_movie_blurred[0,:,:], actin_movie_blurred[1,:,:]]
    image_list = [actin_movie[0,:,:], actin_movie[1,:,:]]
    image_stack = dic.image_stack_from_list(image_list)
    # image_stack = dic.image_stack_from_folder(path,file_type=".tif")
    # skiplist = list(np.arange(3,51,1))
    # skiplist = [int(item) for item in skiplist]
    # image_stack.skip_images(skiplist)
    mesher = dic.Mesher(deg_e = 2, deg_n = 2)
    
    mesh = mesher.mesh(image_stack, GUI = False, n_elx = 10, n_ely = 10, Xc1 = 50, Xc2=actin_movie.shape[2] -50 , Yc1 = 50, Yc2 = actin_movie.shape[1] -50 )
    # mesh = mesher.mesh(image_stack)
    inputs = dic.DICInput(mesh, image_stack)
    inputs.maxit = 2000
    dic_job = dic.DICAnalysis(inputs)
    results = dic_job.run()
    fields = dic.Fields(results)
    displacement_field = fields.disp()
    coordinates = fields.coords()
    
    v_x = np.zeros((displacement_field.shape[-1], displacement_field.shape[2], displacement_field.shape[3]))
    v_y = np.zeros_like(v_x)
    x_coords = np.zeros_like(v_x)
    y_coords = np.zeros_like(v_y)
    x_start_coords = coordinates[0,0,:,:,0]*delta_x
    y_start_coords = coordinates[0,1,:,:,0]*delta_x
    for frame_index in range(displacement_field.shape[-1]):
        v_x[frame_index,:,:] = displacement_field[0,0,:,:,frame_index]*delta_x/delta_t
        v_y[frame_index,:,:] = displacement_field[0,1,:,:,frame_index]*delta_x/delta_t
        x_coords[frame_index,:,:] = coordinates[0,0,:,:,frame_index]*delta_x
        y_coords[frame_index,:,:] = coordinates[0,1,:,:,frame_index]*delta_x
    
    fig = plt.figure(figsize = (2.5,2.5), constrained_layout = True)
    def animate(frame_index): 
        plt.cla()
        optical_flow.costum_imshow(actin_movie[frame_index,:,:], delta_x = delta_x, autoscale=False)
        plt.scatter(x_coords[frame_index] + v_x[frame_index]*delta_t, y_coords[frame_index] + v_y[frame_index]*delta_t)
        # plt.quiver( x_start_coords[:,:],y_start_coords[:,:], v_x[i,:,:],-v_y[i,:,:], color = 'magenta',headwidth=5, scale = 1.0)#quiver([X,Y],U,V,[C])#arrow is in wrong direction because matplt and quiver have different coordanites
            
    ani = FuncAnimation(fig, animate, frames=v_x.shape[0])
    #ani.save('Visualizing Velocity.gif')
    saving_name = os.path.join(os.path.dirname(__file__),'output', 'mu_dic_visualisation.mp4')
    ani.save(saving_name,dpi=600) 
    

if __name__ == '__main__':
    try_mu_dic()