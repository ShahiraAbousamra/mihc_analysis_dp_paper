import os
import numpy as np
from PIL import Image
from scipy.ndimage import filters
from skimage.segmentation import slic
import skimage.color as color
import glob
import cv2
import sys

# Map each label text to RGB color
#text_2_rgb_id = {
#    'cd16': [(0, 0, 0), 2, (0, 0, 1, 0, 0, 0, 0, 0)],
#    'cd20': [(255, 0, 0), 5, (0, 0, 0, 0, 0, 1, 0, 0)],
#    'cd3': [(255, 255, 0), 4, (0, 0, 0, 0, 1, 0, 0, 0)],
#    'cd4': [(10, 150, 255), 3, (0, 0, 0, 1, 0, 0, 0, 0)],
#    'cd8': [(200, 0, 200), 1, (0, 1, 0, 0, 0, 0, 0, 0)],
#    'k17': [(100, 80, 80), 0, (1, 0, 0, 0, 0, 0, 0, 0)],
#    'k17 neg': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17 - neg': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17-neg': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17-negative': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17 negative tumor': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17-negative tumor': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17-neg tumor': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'k17-neh': [(170, 170, 170), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#    'background': [(0, 100, 0), 7, (0, 0, 0, 0, 0, 0, 0, 1)],
#}

text_2_rgb_id = {
    'cd16': [(0, 0, 0), 2, (0, 0, 1, 0, 0, 0, 0, 0)],
    'cd20': [(255, 0, 0), 5, (0, 0, 0, 0, 0, 1, 0, 0)],
    'cd3': [(255, 255, 0), 4, (0, 0, 0, 0, 1, 0, 0, 0)],
    'cd4': [(10, 150, 255), 3, (0, 0, 0, 1, 0, 0, 0, 0)],
    'cd8': [(200, 0, 200), 1, (0, 1, 0, 0, 0, 0, 0, 0)],
    'k17': [(100, 80, 80), 0, (1, 0, 0, 0, 0, 0, 0, 0)],
    'k17+ tumor': [(100, 80, 80), 0, (1, 0, 0, 0, 0, 0, 0, 0)],
    'k17+': [(100, 80, 80), 0, (1, 0, 0, 0, 0, 0, 0, 0)],
    'k17 tumor': [(100, 80, 80), 0, (1, 0, 0, 0, 0, 0, 0, 0)],
    'background': [(0, 100, 0), 8, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17-': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17 negative tumor': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17-neg tumor': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17-neg': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17- tumor': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
    'k17 neg': [(170, 170, 170), 6, (0, 0, 0, 0, 0, 0, 0, 1)],
}

# Resize patches according to this factor
#patch_resize_factor = 1
patch_resize_factor = 0.5

# Extract patches of this size from .tif tiles
patch_extract_dim = 400

# Patch extraction step size
patch_extract_step_size = 300

# Output patch folder
output_folder = '/nfs/data04/shared/shahira/Multiplex_datasets/Multiplex_dots_final_vis'

# Number of stains
n_stains = 7
fix_hematoxilin = False
if(fix_hematoxilin ):
    n_stains += 1

# Load a single point file. Such as:
# L6745/Image_645.tif-points/Points 1.txt
def load_point(point_file_path):
    lines = [line.rstrip() for line in open(point_file_path)]

    # First line, second field
    label_text = lines[0].split('\t')[1].lower()
    if label_text not in text_2_rgb_id:
        print('****************')
        print('label_text no match found:',label_text)
        print('****************')
        return None, None
    label_rgb_id = text_2_rgb_id[label_text]

    xy_coords = np.array([[float(f) * patch_resize_factor for f in x.split()] for x in lines[3:]])

    return label_rgb_id, xy_coords

# Break image and labels to tiles
def break_to_patches(tif_path, im, im_labelvis, im_labelmat, im_labelmat_conc):
    output_prefix = os.path.split(tif_path)[1][:-len('.tif')].replace('/', '-').replace('_', '-')

    for x in list(range(0, im.shape[0] - patch_extract_dim, patch_extract_step_size)) + [im.shape[0] - patch_extract_dim]:
        for y in list(range(0, im.shape[1] - patch_extract_dim, patch_extract_step_size)) + [im.shape[1] - patch_extract_dim]:
            patch = im[x : x + patch_extract_dim, y : y + patch_extract_dim, :]
            Image.fromarray(patch).save('{}/DotsIM-{}-{}-{}.png'.format(output_folder, output_prefix, x, y))

            patch_labelmat = im_labelmat[x : x + patch_extract_dim, y : y + patch_extract_dim]
            patch_labelmat_conc = im_labelmat_conc[x : x + patch_extract_dim, y : y + patch_extract_dim]
            training_patch = np.concatenate((patch, patch_labelmat[..., np.newaxis], patch_labelmat_conc), axis=-1)
            # shahira: fix hematoxilin
            if(fix_hematoxilin):
                hue = color.rgb2hsv(training_patch[:,:,0:3] )[:,:,0]*240;
                training_patch[(hue > 130) & (hue<160) & (training_patch[:,:,4:-1].sum(axis=-1)==0) & (abs(training_patch[:,:,2]-training_patch[:,:,0])>10),10] = 1
                training_patch[training_patch[:,:,10]>0,11]=0
            np.save('{}/Dots-Conc{}-{}-{}.npy'.format(
                output_folder, output_prefix, x, y,                
                ), training_patch)

            patch_labelvis = im_labelvis[x : x + patch_extract_dim, y : y + patch_extract_dim, :]
            patch_labelvis[training_patch[:,:,10]>0] = (120,150,170);
            print('H sum = ', (training_patch[:,:,10]>0).sum());
            Image.fromarray(patch_labelvis).save('{}/DotsVis-{}-{}-{}.png'.format(output_folder, output_prefix, x, y))


# Process each tif and point folder. Such as:
# L6745/Image_645.tif L6745/Image_645.tif-points
def process_tif_and_points(tif_path, points_folder_path, do_break_to_patches=True):
    print('Doing', tif_path, points_folder_path)
    
    # The multiplex image
    im = Image.open(tif_path).convert('RGB')
    im = im.resize((int(im.size[0] * patch_resize_factor), int(im.size[1] * patch_resize_factor)), Image.BICUBIC)
    im = np.array(im)
    print('im',im.shape)
    im2 = np.copy(im)
    # The label image for visualization
    im_labelvis = np.ones_like(im) * 255
    im_labelvis_points = np.ones_like(im) * 255
    # The label matrix
    im_labelmat = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    im_labelmat_conc = np.zeros((im.shape[0], im.shape[1], n_stains), dtype=np.float)
    im_labelmat_points = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)

    # SLIC superpixel
    im_segments = slic(im, compactness = 10.0, n_segments = im.shape[0] * im.shape[1] / 200, sigma = 5)

    for point_file in os.listdir(points_folder_path):
        if not point_file.endswith('.txt'):
            continue
        point_file_path = os.path.join(points_folder_path, point_file)
        label_rgb_id, xy_coords = load_point(point_file_path)
        if label_rgb_id == None or len(xy_coords) == 0:
            continue

        label_rgb, label_id, label_conc = label_rgb_id
        for x, y in xy_coords:
            int_x = int(x)
            int_y = int(y)

            if(int_y < 0):
                int_y = 0
            if(int_x < 0):
                int_x = 0
            if(int_y >= im.shape[0]):
                int_y = im.shape[0]-1
            if(int_x >= im.shape[1]):
                int_x = im.shape[1]-1
            seg_id = im_segments[int_y, int_x]
            im_labelvis[im_segments == seg_id, :] = label_rgb
            im_labelmat[im_segments == seg_id] = label_id
            im_labelmat_conc[im_segments == seg_id,:] = label_conc

            im2=cv2.drawMarker(im2, (int_x, int_y),  color=(255,255,255),markerType= cv2.MARKER_CROSS,markerSize=10,            thickness=4)
            im2=cv2.drawMarker(im2, (int_x, int_y),  color=label_rgb,markerType= cv2.MARKER_CROSS,markerSize=10,            thickness=2)

            im_labelvis_points[max(0,int_y - 2) : min(im.shape[0],int_y + 3), max(0,int_x - 2) : min(im.shape[1],int_x + 3), :] = label_rgb
            im_labelmat_points[int_y, int_x] = label_id

    #background_xys = (filters.gaussian_filter((im_labelmat_points > 0).astype(np.float32), sigma = 10.0, truncate = 4.0) < 0.001)
    #im_labelvis_points[background_xys, :] = text_2_rgb_id['background'][0]
    #im_labelmat_points[background_xys] = text_2_rgb_id['background'][1]

    #tif_path_split = tif_path.split('/')
    #output_prefix =  tif_path_split[-2] +'-' +tif_path_split[-1][:-len('.tif')]
    output_prefix =  os.path.splitext(os.path.basename(tif_path))[0]
    im_label_path = os.path.join(output_folder, output_prefix + '-label.png')
    #im_anno_path = tif_path[:-len('.tif')] + '_anno.png'
    im_anno_path = os.path.join(output_folder, output_prefix + '-anno.png')
    #im_label_points_path = tif_path[:-len('.tif')] + '_label_points.png'
    im_label_points_path = os.path.join(output_folder, output_prefix + '-label_points.png')
    im_path = os.path.join(output_folder, output_prefix + '-im.png')
    Image.fromarray(im_labelvis).save(im_label_path)
    Image.fromarray(im_labelvis_points).save(im_label_points_path)
    Image.fromarray(im2).save(im_anno_path)
    Image.fromarray(im).save(im_path)

    image_wlabel = np.concatenate((im, im_labelmat[..., np.newaxis], im_labelmat_conc, im_labelmat_points[..., np.newaxis]), axis=-1)
    np.save('{}/{}-im_superpixel_conc_dots.npy'.format(
        output_folder, output_prefix                
        ), image_wlabel.astype(np.uint8))

    if(do_break_to_patches):
        break_to_patches(tif_path, im, im_labelvis, im_labelmat, im_labelmat_conc)

    

# Break image Only to tiles
def break_image_to_patches(png_path, output_folder):
    im = Image.open(png_path).convert('RGB')
    im = im.resize((int(im.size[0] * patch_resize_factor), int(im.size[1] * patch_resize_factor)), Image.BICUBIC)
    im = np.array(im)

    filename = os.path.splitext(os.path.split(png_path)[1])[0];
    wsi_name = filename[filename.find('_match_')+len('_match_') : filename.find('_wsi')]
    img_name = filename[ : filename.find('_match')]
    output_prefix = wsi_name + '-' + img_name
    output_prefix = output_prefix.replace('_','-' )

    for x in list(range(0, im.shape[0] - patch_extract_dim, patch_extract_step_size)) + [im.shape[0] - patch_extract_dim]:
        for y in list(range(0, im.shape[1] - patch_extract_dim, patch_extract_step_size)) + [im.shape[1] - patch_extract_dim]:
            patch = im[x : x + patch_extract_dim, y : y + patch_extract_dim, :]
            Image.fromarray(patch).save('{}/DotsIM-{}-{}-{}.png'.format(output_folder, output_prefix, x, y))




def main():
    input_dir = "/home/mihc_analysis_dp_paper/datasets/sup_train_patches/raw_dots"
    images_dir = "/home/mihc_analysis_dp_paper/datasets/sup_train_patches/images"
    print('input_dir',input_dir)
    #for folder in os.listdir('./'):
    for folder in os.listdir(input_dir):
        print('folder',folder)
        folder_path = os.path.join(input_dir,folder)
        if not os.path.isdir(folder_path):
            print('folder not dir')
            continue

        for tif in os.listdir(folder_path):
            print('tif',tif)
            if not tif.endswith('tif'):
                print('continue')
                continue
            im_filepath = glob.glob(os.path.join(images_dir, folder.split('_')[0]+'*'+os.path.splitext(tif)[0]+'*_wsi.png'))
            if(len(im_filepath)==0):
                print('registered image not found:',folder,tif)
                print('continue')
                continue 
            im_filepath = im_filepath[0]           
            tif_path = os.path.join(folder_path, tif)
            points_folder_path = tif_path + '-points'
            print('points_folder_path',points_folder_path)
            if not os.path.isdir(points_folder_path):
                points_folder_path = os.path.join(folder_path, 'unzipped', os.path.splitext(tif)[0])
                print('points_folder_path',points_folder_path)
                if not os.path.isdir(points_folder_path):
                    print('continue')
                    continue

            process_tif_and_points(im_filepath, points_folder_path, do_break_to_patches=False)
            sys.stdout.flush()
        #break

#def main():
#    img_dir = '/gpfs/home/sabousamra/60_registered_no_k17_neg';
#    label_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex/data_and_code_from_le_build';
#    img_files = glob.glob(os.path.join(img_dir, '*wsi.png'))
#    for file in img_files:
#        filename = os.path.split(file)[1];
#        wsi_name = filename[:filename.find('-multires_')];
#        img_name = filename[filename.find('-multires_') + len('-multires_'):filename.find('_wsi')];
#        points_folder_path = os.path.join(label_dir, wsi_name, img_name + '.tif-points');
#        print(file);
#        print(wsi_name);
#        print(img_name);
#        print(points_folder_path);
#        print('');
#        if not os.path.isdir(points_folder_path):
#            continue
#        print('found');

#        process_tif_and_points(file, points_folder_path)


if __name__ == '__main__':
    main()

    #files = glob.glob('/gpfs/projects/KurcGroup/sabousamra/multiplex/registration/output_patches/*.png');
    #output_folder = '/gpfs/projects/KurcGroup/sabousamra/multiplex/registration/output_patches_tiled';
    #for file in files:
    #    break_image_to_patches(file, output_folder);
    

