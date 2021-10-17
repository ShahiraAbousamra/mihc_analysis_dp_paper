import os
import numpy as np
from skimage import io;
import glob;
#from skimage.draw import circle_perimeter
#from skimage.measure import label, find_contours
import cv2 ; #############
#import scipy;
import sys;
from scipy.misc import imresize
from scipy.ndimage.filters import convolve    
from skimage.measure import label
import pickle;
#import openslide
import matplotlib.pyplot as plt;
from skimage import filters

# text_2_rgb_id[stain_name][1]: stain index in my output
# text_2_rgb_id[stain_name][2]: stain one hot encoding in my output
# text_2_rgb_id[stain_name][3]: stain rgb in ref stains

text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30)],
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62)],
    'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151)],
    'k17 neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17 - neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17-neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17-negative': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17 negative tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17-negative tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17-neg tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'k17-neh': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
}

mz_indx_dict = {'cd3':3, 'cd8':5,'cd16':1, 'cd4':4, 'cd20':2, 'k17':6, 'background':7}



# Resize patches according to this factor
patch_resize_factor = 1
# Extract patches of this size from .tif tiles
patch_extract_dim = -1
# Patch extraction step size
patch_extract_step_size = -1
# NN resize factor
NN_resize_factor = 1;

def transform_intensity_to_optical_density(img_rgb, const_val=255.0):  
    #od = -np.log((img_rgb+1)/255.0); 
    if(not isinstance(img_rgb , np.ndarray)):
        img_rgb = np.array(img_rgb);
    img_rgb[np.where(img_rgb <5)] = 5;
    od = -np.log((img_rgb)/const_val); 
    #print(od.shape);
    return od ;

def transform_optical_density_to_intensity(od, const_val=255.0):    
    #print('transform_optical_density_to_intensity')
    if(not isinstance(od , np.ndarray)):
        od = np.array(od);
    rgb = np.exp(-od)*const_val ###
    return rgb ;






def create_seg_map(filepath, out_dir, conc_thresh_dict, size_thresh_dict, resize_ratio, do_thresh_size=False, do_fill_holes=False, do_mz_map=False, do_dilate=False):
    if(os.path.isfile(os.path.join(os.path.splitext(filepath)[0]+'.png'))):
        img = io.imread(os.path.join(os.path.splitext(filepath)[0]+'.png'));
    if(not os.path.isfile(os.path.join(os.path.splitext(filepath)[0]+'.npy'))):
        return;
    conc_arr = np.load(os.path.join(os.path.splitext(filepath)[0]+'.npy'));    
    out_filepath_argmax = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_argmax.npy');
    out_filepath_argmax_mz= os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_argmax_mz.npy');
    if(os.path.isfile(out_filepath_argmax)):
        return;
    out_filepath_conc = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_conc.npy');
    out_filepath_img = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_seg.png');

    conc_arr_new = []
    kernel=np.ones((3,3))

    # resize to original size
    for i in range(conc_arr.shape[0]):
        if(resize_ratio != 1):
            conc_arr_new.append(imresize(conc_arr[i], (int(img.shape[0]*resize_ratio), int(img.shape[1]*resize_ratio)), interp ='bicubic' , mode='F'));
        else:
            conc_arr_new.append(conc_arr[i]);
        #conc_arr_new.append(cv2.resize(cv2.UMat(conc_arr[i]), None, fx=img.shape[0]*resize_ratio, fy=img.shape[1]*resize_ratio,interpolation=cv2.INTER_CUBIC));
    # apply threshold
    for stain_name, thresh in conc_thresh_dict.items():
        stain_idx = text_2_rgb_id[stain_name][1];
        conc_arr_new[stain_idx][np.where(conc_arr_new[stain_idx] < thresh)] = 0;
    conc_arr_new = np.stack(conc_arr_new, axis=0)
    print('------------------------')
    print(conc_arr_new.shape)

    # none_arr will hold mask for areas that have none of the stains since we are now using the background as K17-neg
    # it is concatenated at the beginning of the conc array
    none_arr = (conc_arr_new.sum(axis=0)==0).astype(np.uint8);    
    none_arr = np.expand_dims(none_arr, axis=0);
    conc_arr_new = np.concatenate((none_arr, conc_arr_new), axis=0)
    print('conc_arr.shape = ', conc_arr_new.shape)

    # get argmax
    conc_arr_argmax = conc_arr_new.argmax(axis=0);


    if(do_thresh_size or do_fill_holes):
        # get rid of small detections and fill holes
        conc_arr_argmax_new = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
        for i in range(len(stain_names)):
            binary_mask = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            #binary_mask_out = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1],3)).astype(np.uint8)
            binary_mask_out = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            stain_idx = text_2_rgb_id[stain_names[i]][1] + 1;
            binary_mask[conc_arr_argmax == stain_idx]=255
            #if(do_fill_holes):
            #    poly = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #else:
            #    poly = cv2.findContours(binary_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)            
            poly = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours,hia = poly
            except:
                im2, contours, hierarchy = poly;
            for idx in range(len(contours)):
                contour_i = contours[idx]
                physical_size = cv2.contourArea(contour_i)
                #print(physical_size)
                ##if physical_size>4000 or physical_size<85:
                #if do_thresh_size and physical_size<21:
                ##if do_thresh_size and physical_size<85:
                if do_thresh_size and physical_size<size_thresh_dict[stain_names[i]]:
                    continue
                cv2.drawContours(binary_mask_out, contours, idx, 255, -1)
            print('do_dilate',do_dilate)
            if(do_dilate):
                binary_mask_out = cv2.dilate(binary_mask_out, kernel, iterations=1)
            conc_arr_argmax_new[np.where(binary_mask_out == 255)] = stain_idx
            #cv2.imwrite(stain_names[i]+'.png',binary_mask_out)
        conc_arr_argmax = conc_arr_argmax_new;



    conc_arr_argmax.astype(np.uint8).dump(out_filepath_argmax)

    if(do_mz_map):
        # convert to mz format
        conc_arr_argmax_mz = conc_arr_argmax.copy()
        #conc_arr_argmax_mz[np.where(conc_arr_argmax == 0)] = 8;
        for stain_name, indx_mz in mz_indx_dict.items():
            indx_sh = text_2_rgb_id[stain_name][1] + 1;
            conc_arr_argmax_mz[np.where(conc_arr_argmax == indx_sh)] = indx_mz
        conc_arr_argmax_mz.astype(np.uint8).dump(out_filepath_argmax_mz);

    ## get new conc map
    #conc_arr_flatten = conc_arr_new.reshape((conc_arr_new.shape[0], conc_arr_new.shape[1]*conc_arr_new.shape[2]));
    #conc_arr_flatten_argmax = conc_arr_flatten.argmax(axis=0);
    #one_hot = np.zeros(conc_arr_flatten.shape);
    #one_hot[conc_arr_flatten_argmax,np.arange(one_hot.shape[1])]=1;
    #new_conc_arr_flatten = one_hot * conc_arr_flatten;
    #new_conc_arr = new_conc_arr_flatten.reshape((conc_arr_new.shape[0], conc_arr_new.shape[1], conc_arr_new.shape[2])); 
    #new_conc_arr.astype(np.float16).dump(out_filepath_conc)

    # visualize output
    #print(conc_arr.min())
    #print(conc_arr.max())
    #print(new_conc_arr.min())
    #print(new_conc_arr.max())
    #new_out_od = [];
    #stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin', 'background'];
    #for i in range(len(stain_names)):
    #    stain_name = stain_names[i];
    #    stain_rgb = text_2_rgb_id[stain_name][3];
    #    print(stain_rgb)
    #    stain_od = transform_intensity_to_optical_density(stain_rgb);
    #    print(stain_od)
    #    stain_idx = text_2_rgb_id[stain_name][1];
    #    stain_conc_arr = new_conc_arr[stain_idx];
    #    #stain_conc_arr[np.where(stain_conc_arr < thresh)] = 0;
    #    stain_conc_img_od = np.matmul(stain_od.reshape((-1,1)), stain_conc_arr.reshape(1,-1));
    #    new_out_od.append(stain_conc_img_od );

    #stain_conc_img_od = new_out_od[0];
    #for i in range(1, len(new_out_od)):
    #    stain_conc_img_od = stain_conc_img_od + new_out_od[i];
            

    #stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
    #stain_conc_img_od = stain_conc_img_od.reshape((conc_arr_new.shape[1],conc_arr_new.shape[2],3))
    #stain_conc_img_rgb = transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
    #io.imsave(out_filepath_img, stain_conc_img_rgb);

    return;




if __name__ == '__main__':

    #stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin', 'background'];
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin'];

    size_thresh_dict = {'cd3':21, 'cd4':21, 'cd8':21, 'cd16':21, 'cd20':21, 'k17':21, 'hematoxilin':21, 'background':21}
    #size_thresh_dict = {'cd3':84, 'cd4':84, 'cd8':84, 'cd16':84, 'cd20':84, 'k17':84, 'hematoxilin':84, 'background':84}

    resize_ratio = 1
    ################################################################################################################################
    ######## test 19 patches data #########
    root = '/gpfs/projects/KurcGroup/sabousamra/multiplex1.0/eval_19patches'

    ######## supervised noscale #########
    conc_dir_name = 'sup_dot_wsi2_noscale_19patches'
    thresh_dict = {'cd3':0.4, 'cd4':0.4, 'cd8':0.5, 'cd16':0.4, 'cd20':0.3, 'k17':0.5, 'hematoxilin':0.1, 'background':1.7}

    ######## Processing #########
    in_dir = os.path.join(root, conc_dir_name)
    out_dir = os.path.join(root, conc_dir_name + '-a_conc-th_size-th_fill_dil')
    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    files = glob.glob(in_dir+'/*.npy') 
    for file in files:
        print(file)
        create_seg_map(file, out_dir, thresh_dict, size_thresh_dict, resize_ratio, do_thresh_size=True, do_fill_holes=True, do_mz_map=False, do_dilate=True) # sup
        sys.stdout.flush()

