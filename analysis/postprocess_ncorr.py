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
    v_x = np.zeros((len(ncorr_result['data_dic_save']['displacements']),1024,1024))
    v_y = np.zeros((len(ncorr_result['data_dic_save']['displacements']),1024,1024))
    for frame_index, displacement_dict in enumerate(ncorr_result['data_dic_save']['displacements']):
        # v_x[frame_index,:,:] = displacement_dict['plot_u_cur_formatted']*delta_x/delta_t
        print(frame_index)
        correlation_coefficients = ncorr_result['data_dic_save']['displacements'][frame_index]['plot_corrcoef_dic']
        v_x[frame_index,:,:] = displacement_dict['plot_u_dic']*delta_x/delta_t
        v_y[frame_index,:,:] = displacement_dict['plot_v_dic']*delta_x/delta_t
        v_x[frame_index,:,:][correlation_coefficients<0.3] = 0.0
        v_y[frame_index,:,:][correlation_coefficients<0.3] = 0.0
        # import pdb; pdb.set_trace()
        # v_y[frame_index,:,:] = displacement_dict['plot_v_cur_formatted']*delta_x/delta_t
        # print(displacement_dict['plot_u_cur_formatted'].shape)
        # print(np.mean(displacement_dict['plot_u_cur_formatted']))
        # print(np.max(displacement_dict['plot_u_cur_formatted']))


    return v_x, v_y

def visualise_ncorr_result():
    ncorr_result = mat73.loadmat((os.path.join(os.path.dirname(__file__), 'data', 'ncorr_try_actin_large_blurred_again.mat')))

    # ncorr_result = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'data', 'ncorr_first_frame_part.mat'))
    v_x, v_y = convert_ncorr_result(ncorr_result)
    speed = np.sqrt(v_x**2 + v_y**2)

    # actin_movie = skimage.io.imread(os.path.join(os.path.dirname(__file__),'data','LifeActin-Ruby_MB160918_20_a_control.tif'))
    # Replace 'path/to/your/tiff/images' with the actual path to your images
    image_collection = skimage.io.imread_collection(os.path.join(os.path.dirname(__file__),'data','actin_image_sequence_large','*.tif'))
    actin_movie = image_collection.concatenate()

    
    flow_result = dict()
    flow_result['v_x'] = v_x
    flow_result['v_y'] = v_y
    flow_result['original_data'] = actin_movie[:7,:,:]
    flow_result['delta_x'] = delta_x
    # print(np.sum(v_y>0)/279*1/379)

    optical_flow.make_velocity_overlay_movie(flow_result,
                                             os.path.join(os.path.dirname(__file__),'output','ncorr_velocities.mp4'), 
                                             arrow_scale = 0.3,
                                             arrow_boxsize = 30)


def plot_corr_coeffs_first_two_frames():
    ncorr_result = mat73.loadmat((os.path.join(os.path.dirname(__file__), 'data', 'ncorr_try_actin_large_blurred_again.mat')))
    corr_coeffs1 = ncorr_result['data_dic_save']['displacements'][0]['plot_corrcoef_dic']
    corr_coeffs2 = ncorr_result['data_dic_save']['displacements'][1]['plot_corrcoef_dic']
    all_corr_coeffs = np.array(corr_coeffs1.flatten().tolist() + corr_coeffs2.flatten().tolist())
    
    print(np.min(all_corr_coeffs[all_corr_coeffs>0]))
    
    plt.figure()
    plt.hist(all_corr_coeffs, bins = 500)
    plt.axvline(0.02, color = 'black')
    plt.xlabel('correlation coefficient')
    plt.ylabel('bincount')
    # plt.xlim(0.0,0.025)
    plt.yscale('log')
    # plt.ylim(0,100)
    plt.savefig(os.path.join(os.path.dirname(__file__),'output','ncorr_correlation_coefficients.pdf'))
    

if __name__ == '__main__':
    visualise_ncorr_result()
    # plot_corr_coeffs_first_two_frames()