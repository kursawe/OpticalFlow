import scipy.io
import os
import compare_rho_and_actin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import skimage
import numpy as np
import mat73

sys.path.append(os.path.join(os.path.dirname(__file__),'..','source'))
import optical_flow


delta_x = compare_rho_and_actin.delta_x
delta_t = compare_rho_and_actin.delta_t

def convert_ncorr_result(ncorr_result):
    print(ncorr_result.keys())
    # print(ncorr_result['data_dic_save'])
    # print(ncorr_result['reference_save'])
    # print(ncorr_result['current_save'])
    print(ncorr_result['data_dic_save']['displacements'][1].keys())
    v_x = np.zeros((len(ncorr_result['data_dic_save']['displacements']),379,279))
    v_y = np.zeros((len(ncorr_result['data_dic_save']['displacements']),379,279))
    for frame_index, displacement_dict in enumerate(ncorr_result['data_dic_save']['displacements']):
        # v_x[frame_index,:,:] = displacement_dict['plot_u_cur_formatted']*delta_x/delta_t
        print(frame_index)
        v_x[frame_index,:,:] = displacement_dict['plot_u_dic']*delta_x/delta_t
        v_y[frame_index,:,:] = displacement_dict['plot_v_dic']*delta_x/delta_t
        # v_y[frame_index,:,:] = displacement_dict['plot_v_cur_formatted']*delta_x/delta_t
        # print(displacement_dict['plot_u_cur_formatted'].shape)
        # print(np.mean(displacement_dict['plot_u_cur_formatted']))
        # print(np.max(displacement_dict['plot_u_cur_formatted']))


    return v_x, v_y

def visualise_ncorr_result():
    ncorr_result = mat73.loadmat((os.path.join(os.path.dirname(__file__), 'data', 'ncorr_try.mat')))

    # ncorr_result = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'data', 'ncorr_first_frame_part.mat'))
    v_x, v_y = convert_ncorr_result(ncorr_result)
    speed = np.sqrt(v_x**2 + v_y**2)

    actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    
    flow_result = dict()
    flow_result['v_x'] = v_x
    flow_result['v_y'] = v_y
    flow_result['original_data'] = actin_movie[:3,:,:]
    flow_result['delta_x'] = delta_x
    print(np.sum(v_y>0)/279*1/379)

    optical_flow.make_velocity_overlay_movie(flow_result,
                                             os.path.join(os.path.dirname(__file__),'output','ncorr_velocities.mp4'), 
                                            #  arrow_scale = 1.0,
                                             arrow_boxsize = 15)


if __name__ == '__main__':
    visualise_ncorr_result()