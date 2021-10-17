import numpy as np
import os
import glob
from skimage.measure import label
import skimage.io as io
import cv2

# Create different ensemble from 2 predictions (Union, intersection, anchored-union).

text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30)],
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62)],
    #'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151)],
    #'k17 neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17 - neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17-neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17-negative': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17 negative tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17-negative tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17-neg tumor': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'k17-neh': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175)],
    #'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
}



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

def visualize_contour_from_multichannel(pred_filepath, out_dir, img_filepath):
    out_filepath = os.path.join(out_dir, os.path.splitext(os.path.split(pred_filepath)[1])[0] + '_mask' + '.png');

    #pred_ext = os.path.splitext(argmax_filepath)[1];
    #if(pred_ext == '.png'):
    #    conc_arr_argmax = io.imread(argmax_filepath);
    #else:
    #    conc_arr_argmax = np.load(argmax_filepath);
    pred = np.load(pred_filepath)
    
    # resize to original size
    img = io.imread(img_filepath);
    #conc_arr_argmax = imresize(conc_arr_argmax,(int(img.shape[0]),int(img.shape[1])),interp='nearest', mode='F').astype(np.uint8);
    img = img[:pred.shape[1], :pred.shape[2], :]


    ##conc_arr_argmax = conc_arr.argmax(axis=0);
    #conc_arr_flatten = np.ones((9, conc_arr_argmax.shape[0]*conc_arr_argmax.shape[1]));
    ##conc_arr_flatten_argmax = conc_arr_flatten.argmax(axis=0);
    #conc_arr_flatten_argmax = conc_arr_argmax.reshape((-1));
    #one_hot = np.zeros(conc_arr_flatten.shape);
    #one_hot[conc_arr_flatten_argmax,np.arange(one_hot.shape[1])]=1;
    ##new_conc_arr_flatten = one_hot * conc_arr_flatten;
    #new_conc_arr_flatten = one_hot ;
    #new_conc_arr = new_conc_arr_flatten.reshape((9, conc_arr_argmax.shape[0], conc_arr_argmax.shape[1]));
    new_out_od = [];

    #stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin', 'background'];
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17'];
    for i in range(len(stain_names)):
        stain_name = stain_names[i];
        stain_rgb = text_2_rgb_id[stain_name][3];
        print(stain_rgb)
        stain_od = transform_intensity_to_optical_density(stain_rgb);
        print(stain_od)
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        #if(argmax_ext == '.png'):
        #     stain_idx = mz_indx_dict[stain_name]
        stain_conc_arr = pred[stain_idx];
        #stain_conc_arr[np.where(stain_conc_arr < thresh)] = 0;
        stain_conc_img_od = np.matmul(stain_od.reshape((-1,1)), stain_conc_arr.reshape(1,-1));
        new_out_od.append(stain_conc_img_od );

    stain_conc_img_od = new_out_od[0];
    for i in range(1, len(new_out_od)):
        stain_conc_img_od = stain_conc_img_od + new_out_od[i];
            

    stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
    stain_conc_img_od = stain_conc_img_od.reshape((pred.shape[1],pred.shape[2],3))
    stain_conc_img_rgb = transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
    io.imsave(out_filepath, stain_conc_img_rgb);

    #size_thresh = {'cd3':85, 'cd4':0, 'cd8':20, 'cd16':0, 'cd20':0, 'k17':120, 'hematoxilin':0, 'background':0} # unsup
    #size_thresh = {'cd3':85, 'cd4':85, 'cd8':85, 'cd16':85, 'cd20':85, 'k17':85, 'hematoxilin':85, 'background':85} # sup
    size_thresh = {'cd3':0, 'cd4':0, 'cd8':0, 'cd16':0, 'cd20':0, 'k17':0, 'hematoxilin':0, 'background':0} # mz

    if(not (img_filepath is None)):
        im = cv2.imread(img_filepath);
        kernel = np.ones((3,3),np.uint8)
        for i in range(len(stain_names)):
            binary_mask = np.zeros((pred.shape[1],pred.shape[2])).astype(np.uint8)
            im_out = np.copy(im)
            binary_mask_out = np.zeros((pred.shape[1],pred.shape[2])).astype(np.uint8)
            stain_idx = text_2_rgb_id[stain_names[i]][1] + 1;
            #if(argmax_ext == '.png'):
            #     stain_idx = mz_indx_dict[stain_names[i]]
            binary_mask[pred[stain_idx] == 1]=255
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.split(pred_filepath)[1])[0] + '_binary_exact_mask_' + stain_names[i] + '_s'+str(size_thresh[stain_names[i]])+'.png'), (binary_mask).astype(np.uint8))            
            #poly = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            poly = cv2.findContours(binary_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours,hia = poly
            except:
                im2, contours, hierarchy = poly;
            for idx in range(len(contours)):
                contour_i = contours[idx]
                physical_size = cv2.contourArea(contour_i)
                #print(physical_size)
                #if physical_size>4000 or physical_size<85:
                if physical_size<size_thresh[stain_names[i]]:
                    continue
                #cv2.drawContours(binary_mask_out, contours, idx, 255, 2)
                cv2.drawContours(binary_mask_out, contours, idx, 255, -1)
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.split(pred_filepath)[1])[0] + '_binary_mask_' + stain_names[i] + '_s'+str(size_thresh[stain_names[i]])+'.png'), (binary_mask_out).astype(np.uint8))            
            binary_mask_out = cv2.dilate(binary_mask_out,kernel,iterations = 1)
            #poly = cv2.findContours(binary_mask_out, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            poly = cv2.findContours(binary_mask_out, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours,hia = poly
            except:
                im2, contours, hierarchy = poly;
            #cv2.drawContours(im_out, contours, -1, (0,255,255), 2)
            cv2.drawContours(im_out, contours, -1, (0,0,0), 2)
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.split(pred_filepath)[1])[0] + '_contour_' + stain_names[i] + '_s'+str(size_thresh[stain_names[i]])+'.png'), im_out)

    return;


def combine_pred_intersection(unet_argmax_filepath, ae_argmax_filepath, out_filename):
    unet_argmax = np.load(unet_argmax_filepath) 
    ae_argmax = np.load(ae_argmax_filepath) 
    min_height = min(unet_argmax.shape[0], ae_argmax.shape[0])
    min_width = min(unet_argmax.shape[1], ae_argmax.shape[1])
    unet_argmax = unet_argmax[:min_height, :min_width]
    ae_argmax = ae_argmax[:min_height, :min_width]
    combined_pred = np.zeros((len(text_2_rgb_id.keys())+1, min_height, min_width))
    
    for stain_name in text_2_rgb_id.keys():
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        unet_binary_map = (unet_argmax == stain_idx)
        ae_binary_map = (ae_argmax == stain_idx)
        combined_pred[stain_idx, :,:] = (unet_binary_map * ae_binary_map)

    #combined_pred[0,:,:] = (1-combined_pred.sum(axis=0)).astype(np.uint8)
    combined_pred[0,:,:] = (1-combined_pred.sum(axis=0))
    combined_pred[np.where(combined_pred < 0)] = 0

    combined_pred.astype(np.uint8).dump(out_filename)

def combine_pred_union(unet_argmax_filepath, ae_argmax_filepath, out_filename):
    unet_argmax = np.load(unet_argmax_filepath) 
    ae_argmax = np.load(ae_argmax_filepath) 
    min_height = min(unet_argmax.shape[0], ae_argmax.shape[0])
    min_width = min(unet_argmax.shape[1], ae_argmax.shape[1])
    unet_argmax = unet_argmax[:min_height, :min_width]
    ae_argmax = ae_argmax[:min_height, :min_width]
    combined_pred = np.zeros((len(text_2_rgb_id.keys())+1, min_height, min_width))
    
    for stain_name in text_2_rgb_id.keys():
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        unet_binary_map = (unet_argmax == stain_idx)
        ae_binary_map = (ae_argmax == stain_idx)
        combined_pred[stain_idx, :,:] = (np.logical_or(unet_binary_map , ae_binary_map)).astype(np.uint8)

    #combined_pred[0,:,:] = (1-combined_pred.sum(axis=0)).astype(np.uint8)
    combined_pred[0,:,:] = (1-combined_pred.sum(axis=0))
    combined_pred[np.where(combined_pred < 0)] = 0

    combined_pred.astype(np.uint8).dump(out_filename)

def combine_pred_anchor(argmax_filepath_a, argmax_filepath_b, out_filename):
    argmax_a = np.load(argmax_filepath_a) 
    argmax_b = np.load(argmax_filepath_b) 
    min_height = min(argmax_a.shape[0], argmax_b.shape[0])
    min_width = min(argmax_a.shape[1], argmax_b.shape[1])
    argmax_a = argmax_a[:min_height, :min_width]
    argmax_b = argmax_b[:min_height, :min_width]
    combined_pred = np.zeros((len(text_2_rgb_id.keys())+1, min_height, min_width))
    
    for stain_name in text_2_rgb_id.keys():
        print(stain_name)
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        binary_map_a = (argmax_a == stain_idx)
        binary_map_b = (argmax_b == stain_idx)
        #combined_pred[stain_idx, :,:] = (np.logical_or(unet_binary_map , ae_binary_map)).astype(np.uint8)
        pred_comp_mask_a = label(binary_map_a);
        pred_comp_mask_b = label(binary_map_b);
        combined_pred_binary = np.copy(binary_map_a)
        print('len comp_a = ', pred_comp_mask_a.max())
        print('len comp_b = ', pred_comp_mask_b.max())
        for pred_comp_id_a in range(1,pred_comp_mask_a.max()+1): 
            print('pred_comp_id_a=',pred_comp_id_a)
            pred_comp_binary_a = (pred_comp_mask_a == pred_comp_id_a)
            for pred_comp_id_b in range(1,pred_comp_mask_b.max()+1): 
                pred_comp_binary_b = (pred_comp_mask_b == pred_comp_id_b)
                if(pred_comp_binary_b[np.where(pred_comp_binary_a)].sum() > 0):
                    combined_pred_binary[np.where(pred_comp_binary_b)] = 1

        combined_pred[stain_idx, :,:] = combined_pred_binary            

    #combined_pred[0,:,:] = (1-combined_pred.sum(axis=0)).astype(np.uint8)
    combined_pred[0,:,:] = (1-combined_pred.sum(axis=0))
    combined_pred[np.where(combined_pred < 0)] = 0

    combined_pred.astype(np.uint8).dump(out_filename)


# 1. Create ensemble output numpy: intersection/ union/ union_anchor_unet/ ae_unet_union_anchor_ae

unet_argmax_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/unet_argmax'
ae_argmax_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/supervised_ae_argmax'
png_dir = '/home/mihc_analysis_dp_paper/datasets/test_19patches/images'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_intersection'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_unet'
out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_ae'



files = glob.glob(os.path.join(png_dir,'*.png'))

for img_filepath in files:
    filename = os.path.splitext(os.path.basename(img_filepath))[0]
    print('combining', filename)
    unet_argmax_filepath = os.path.join(unet_argmax_dir, filename+'_argmax.npy')
    ae_argmax_filepath = os.path.join(ae_argmax_dir,  filename+'_argmax.npy')
    out_filpath = os.path.join(out_dir, filename+'_multichannel.npy')
    combine_pred_intersection(unet_argmax_filepath, ae_argmax_filepath, out_filpath)
    #combine_pred_union(unet_argmax_filepath, ae_argmax_filepath, out_filpath)
    #combine_pred_anchor(unet_argmax_filepath, ae_argmax_filepath, out_filpath)
    #combine_pred_anchor(ae_argmax_filepath, unet_argmax_filepath, out_filpath)

# 2. Visualize ensemble

png_dir = '/home/mihc_analysis_dp_paper/datasets/test_19patches/images'
#pred_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_intersection'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_intersection_vis'
#pred_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_vis'
#pred_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_unet'
#out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_unet_vis'
pred_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_ae'
out_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_ae_vis'


files = glob.glob(os.path.join(png_dir,'*.png'))
for img_filepath in files:
    filename = os.path.splitext(os.path.basename(img_filepath))[0]
    print('vis', filename)
    pred_filepath = os.path.join(pred_dir, filename+'_multichannel.npy')
    visualize_contour_from_multichannel(pred_filepath, out_dir, img_filepath)
