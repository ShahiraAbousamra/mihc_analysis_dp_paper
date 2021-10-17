import numpy as np
import os
import skimage.io as io
import skimage.draw as draw
import glob
from skimage.measure import label
from scipy.misc import imresize
import sys

# Read a numpy file containing the dot locations for each stain/cell-type 
# For each dot create a circle and find intersections to get the precision, recall, and F-score

text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30), (0, 0, 0,255)],
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75), (255, 0, 0,255)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45), (255, 255, 0,255)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151), (10, 150, 255,255)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151), (200, 0, 200,255)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62), (100, 80, 80,255)],
    #'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151), (255,255,255,255)],
    #'k17-neg': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 1, 0), (97, 134, 175), (170, 170, 170,255)],
    #'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210), (0, 100, 0,255)],
}

argmax_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/new_test_set19_eval/ae_unet_union_anchor_ae/'
lbl_dir = '/home/mihc_analysis_dp_paper/datasets/test_19patches/dot_labels/'
img_dir = '/home/mihc_analysis_dp_paper/datasets/test_19patches/images'
out_dir= '/gpfs/projects/KurcGroup/sabousamra/multiplex/dots_to_superpixels/dot_labels_k17_f1/ae_unet_union_anchor_ae'
mpp = 0.274
diam_microns = 10
radius_pixels = int(diam_microns/2/mpp)

lbl_files = glob.glob(os.path.join(lbl_dir, '*labelconc.npy'));
n=0
f1_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 
f1_n = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 
f1_mean = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 
precision_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0}  # tp / (tp + fp)
recall_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} # tp / (tp + fn)
tp_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 
fn_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 
fp_total = {'cd3':0,'cd4':0,'cd8':0,'cd16':0,'cd20':0, 'k17':0} 

for lbl_file in lbl_files:
    basefilename = os.path.basename(lbl_file)[:-len('_labelconc.npy')]
    lbl_img = np.load(lbl_file)
    img = io.imread(os.path.join(img_dir, basefilename+'.png'))
    argmax =np.load(os.path.join(argmax_dir, basefilename + '_multichannel.npy'))
    if(len(argmax.shape)==2):
        height, width = argmax.shape[0], argmax.shape[1]
    else:
        height, width = argmax.shape[1], argmax.shape[2]
    lbl_img = lbl_img[:height, :width, :]
    n += 1
    print(basefilename)
    sys.stdout.flush();
    for stain_name in text_2_rgb_id.keys():
        print('stain_name=',stain_name)
        sys.stdout.flush();
        out_filename_f1 = os.path.join(out_dir, 'F1_' + stain_name + '.txt');
        out_filename_tp = os.path.join(out_dir, 'TP_' + stain_name + '.txt');
        out_filename_fn = os.path.join(out_dir, 'FN_' + stain_name + '.txt');
        out_filename_fp = os.path.join(out_dir, 'FP_' + stain_name + '.txt');
        out_filename_pred = os.path.join(out_dir, 'all_pred_' + stain_name + '.txt');
        out_file_tp = open(out_filename_tp, 'a+')
        out_file_fn = open(out_filename_fn, 'a+')
        out_file_fp = open(out_filename_fp, 'a+')
        out_file_pred = open(out_filename_pred, 'a+')
        out_file_f1 = open(out_filename_f1, 'a+')
        #mask_val = text_2_rgb_id[stain_name][4]
        mask_val = text_2_rgb_id[stain_name][2]
        stain_indx = text_2_rgb_id[stain_name][1] + 1
        #lbl_mask = (np.all(lbl_img == mask_val, axis=-1))
        lbl_mask = (np.all(lbl_img[:,:,:] == mask_val, axis=-1))
        dots = np.where(lbl_mask==1) 

        #lbl_comp_mask = label(lbl_mask); 
        if(len(argmax.shape)==2):
            pred_mask = (argmax==stain_indx)
        else:
            pred_mask = argmax[stain_indx]
        pred_comp_mask = label(pred_mask);  
        pred_comp_found = np.zeros(pred_comp_mask.max()+1)   
        tp = fn = fp = 0

        print('len(dots)=',len(dots[0]))
        sys.stdout.flush();
        for lbl_comp_id in range(len(dots[0])):
            found = False;
            print('lbl_comp_id=',lbl_comp_id)
            sys.stdout.flush();
            #lbl_comp_binary = (lbl_comp_mask == lbl_comp_id)            
            lbl_comp_binary = np.zeros((height, width))
            dot_y = dots[0][lbl_comp_id]
            dot_x = dots[1][lbl_comp_id]
            rr,cc = draw.circle(dot_y, dot_x, radius_pixels)
            rr[np.where(rr >= lbl_comp_binary.shape[0])]=lbl_comp_binary.shape[0]-1
            cc[np.where(cc >= lbl_comp_binary.shape[1])]=lbl_comp_binary.shape[1]-1
            rr[np.where(rr < 0)]=0
            cc[np.where(cc < 0)]=0
            lbl_comp_binary[rr,cc] = 1
            
            lbl_size_in_pixels = lbl_comp_binary.sum();
            for pred_comp_id in range(1,pred_comp_mask.max()+1):                
                pred_comp_binary = (pred_comp_mask == pred_comp_id)
                intersection = pred_comp_binary[rr,cc].sum()
                #intersection = ((lbl_comp_binary * pred_comp_binary) == True).sum()           
                if(intersection > 0):                    
                    if(not found):
                        found = True
                        out_file_tp.write(basefilename +','+stain_name+','+str(lbl_size_in_pixels))
                    tp += 1
                    pred_comp_found[pred_comp_id] = 1
                    out_file_tp.write(','+str(pred_comp_binary.sum()) +','+str(intersection))
            if(found):
                out_file_tp.write('\n')
            else:
                fn += 1
                out_file_fn.write(basefilename +','+stain_name+','+str(lbl_size_in_pixels) + '\n')
        out_file_tp.close()
        out_file_fn.close()

        for i in range(1, pred_comp_found.shape[0]):
            print('pred_comp_id=',i)
            sys.stdout.flush();
            pred_comp_binary = (pred_comp_mask == i)
            out_file_pred.write(basefilename +','+stain_name+','+str(pred_comp_binary.sum()) +'\n')
            if(pred_comp_found[i] == 0):
                fp += 1
                out_file_fp.write(basefilename +','+stain_name+','+str(pred_comp_binary.sum()) +'\n')

        if(tp+fp == 0):
            precision = 1
        else:
            precision = tp/(tp+fp)
        if(tp+fn == 0) :
            recall = 1
        else:
            recall = tp/(tp+fn)
        
        if(precision+recall == 0):
            f1 = 0
        else:
            f1 = 2*precision*recall/(precision+recall)
        tp_total[stain_name] += tp
        fn_total[stain_name] += fn
        fp_total[stain_name] += fp
        precision_total[stain_name] += precision
        recall_total[stain_name] += recall
        f1_total[stain_name] += f1

        out_file_f1.write(basefilename +','+stain_name+',' +'TP'+',' +str(tp)+',' +'FN'+',' +str(fn)+',' +'FP'+',' +str(fp)+',' +'Precision'+',' +str(precision)+',' +'Recall'+',' +str(recall)+',' +'F1'+',' +str(f1) + '\n')

        out_file_fp.close()
        out_file_pred.close()

        out_file_f1.close()
   
print('tp_total=',tp_total)
print('fn_total=',fn_total)
print('fp_total=',fp_total)
print('precision_total=',precision_total)
print('recall_total=',recall_total)
print('f1_total=',f1_total)
