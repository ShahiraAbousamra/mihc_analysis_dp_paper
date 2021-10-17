import numpy as np;
import torch;


def transform_intensity_to_optical_density(img_rgb, const_val=255.0):  
    #od = -np.log((img_rgb+1)/255.0); 
    if(isinstance(img_rgb , np.ndarray)):
        img_rgb[np.where(img_rgb <5)] = 5;
        od = -np.log((img_rgb)/const_val); 
    elif(isinstance(img_rgb , torch.Tensor)):
        od = -torch.log((img_rgb)/const_val);         
    #print(od.shape);
    return od ;

def transform_optical_density_to_intensity(od, const_val=255.0):    
    #print('transform_optical_density_to_intensity')
    if(isinstance(od , np.ndarray)):
        rgb = np.exp(-od)*const_val ###
    elif(isinstance(od , torch.Tensor)):
        rgb = torch.exp(-od)*const_val ###
    return rgb ;
 
def transform_rgb_to_cmyk(img_rgb):    
    img_cmyk = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4))
    img_rgb_norm = img_rgb / 255.0;
    k = 1 - np.amax(img_rgb_norm, axis=-1);
    k = np.expand_dims(k,axis=2)
    img_cmyk[:,:,0:3] = np.divide((1 - img_rgb_norm - k) , (1-k));
    img_cmyk[:,:,3] = k.reshape((k.shape[0],k.shape[1]))
    return img_cmyk * 100;

def transform_cmyk_to_rgb(img_cmyk):    
    img_rgb = np.zeros((img_cmyk.shape[0], img_cmyk.shape[1], 3))
    if(img_cmyk.max() > 1):
        img_cmyk = img_cmyk/100.0;
    img_cmyk = 1 - img_cmyk;
    img_rgb[:,:,0] = img_cmyk[:,:,0]*img_cmyk[:,:,3] * 255;
    img_rgb[:,:,1] = img_cmyk[:,:,1]*img_cmyk[:,:,3] * 255;
    img_rgb[:,:,2] = img_cmyk[:,:,2]*img_cmyk[:,:,3] * 255;
    return img_rgb;

def transform_cmyk_to_rgb_1D(img_cmyk):    
    img_rgb = np.zeros((img_cmyk.shape[0], 3))
    if(img_cmyk.max() > 1):
        img_cmyk = img_cmyk/100.0;
    img_cmyk = 1 - img_cmyk;
    img_rgb[:,0] = img_cmyk[:,0]*img_cmyk[:,3] * 255;
    img_rgb[:,1] = img_cmyk[:,1]*img_cmyk[:,3] * 255;
    img_rgb[:,2] = img_cmyk[:,2]*img_cmyk[:,3] * 255;
    return img_rgb;


def rgb2hsl(rgb):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation
    # convert r,g,b [0,255] range to [0,1]
    r = rgb[0] / 255.0;
    g = rgb[1] / 255.0;
    b = rgb[2] / 255.0;
    # get the min and max of r,g,b
    max_val = max(max(r, g), b);
    min_val = min(min(r, g), b);
    # lightness is the average of the largest and smallest color components
    lum = (max_val + min_val) / 2.0;
    hue = 0;
    sat = 0;
    if (max_val == min_val): # no saturation
        hue = 0;
        sat = 0;
    else:
        c = max_val - min_val; # chroma
        # saturation is simply the chroma scaled to fill
        # the interval [0, 1] for every combination of hue and lightness
        sat = c / (1 - abs(2 * lum - 1));
        if(max_val == r):
            # hue = (g - b) / c;
            hue = ((g - b) / c) % 6;
            # hue = (g - b) / c + (g < b ? 6 : 0);
        elif(max_val == g):
            hue = (b - r) / c + 2;
        elif(max_val == b):
            hue = (r - g) / c + 4;
    
    #hue = int(hue * 60 + 0.5); # °
    #sat = int(sat * 100 + 0.5); # %
    #lum = int(lum * 100 + 0.5); # %
    # to get paint values
    hue = int(hue * 60 * 240/360 ); # °
    sat = int(sat * 240 ); # %
    lum = int(lum * 240 ); # %
    return [hue, sat, lum];


def img_rgb2hsl(img_rgb):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation
    # convert r,g,b [0,255] range to [0,1]
    img_rgb = img_rgb / 255.0
    # get the min and max of r,g,b
    img_rgb_max = np.max(img_rgb, axis=-1)
    img_rgb_min = np.min(img_rgb, axis=-1)
    # lightness is the average of the largest and smallest color components
    img_hsl = np.zeros(img_rgb.shape);
    img_hsl[:,:,2] = (img_rgb_max + img_rgb_min) / 2.0;
    c = img_rgb_max - img_rgb_min; # chroma
    # saturation is simply the chroma scaled to fill
    # the interval [0, 1] for every combination of hue and lightness
    img_hsl[:,:,1] = np.div(c , (1 - abs(2 * lum - 1)));
    img_hue = np.zeros((img_hsl.shape[0], img_hsl.shape[1]));
    r = np.mod(np.div(img_rgb[:,:,1] - img_rgb[:,:,2], c), 6);
    g = np.div(img_rgb[:,:,2] - img_rgb[:,:,0], c) + 2;
    b = np.div(img_rgb[:,:,0] - img_rgb[:,:,1], c) + 4;
    img_hue[np.where(img_rgb_max == img_rgb[:,:,0])] = r[np.where(img_rgb_max == img_rgb[:,:,0])];
    img_hue[np.where(img_rgb_max == img_rgb[:,:,1])] = g[np.where(img_rgb_max == img_rgb[:,:,0])];
    img_hue[np.where(img_rgb_max == img_rgb[:,:,2])] = b[np.where(img_rgb_max == img_rgb[:,:,0])];
    img_hsl[:,:,0] = img_hue;
    
    #hue = int(hue * 60 + 0.5); # °
    #sat = int(sat * 100 + 0.5); # %
    #lum = int(lum * 100 + 0.5); # %
    # to get paint values
    img_hsl[:,:,0] = int(img_hsl[:,:,0] * 60 * 240/360 ); # °
    img_hsl[:,:,1] = int(img_hsl[:,:,1] * 240 ); # %
    img_hsl[:,:,2] = int(img_hsl[:,:,2] * 240 ); # %
    return img_hsl;

def hsl2rgb(hsl):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation
    # convert hsl [0,255] range to [0,1]
    # When 0 ≤ H < 360, 0 ≤ S ≤ 1 and 0 ≤ L ≤ 1:
    hue = hsl[0] / 240 * 360;
    sat = hsl[1] / 240.0;
    lum = hsl[2] / 240.0;

    #C = (1 - |2L - 1|) × S
    c = (1 - abs(2*lum-1)) * sat
    
    #X = C × (1 - |(H / 60°) mod 2 - 1|)
    x = c * (1 - abs((hue / 60) % 2 - 1))

    #m = L - C/2
    m = lum - c/2.0

    if(hue < 60):
        r = c;
        g = x;
        b = 0;
    elif(hue < 120):
        r = x;
        g = c;
        b = 0;
    elif(hue < 180):
        r = 0;
        g = c;
        b = x;
    elif(hue < 240):
        r = 0;
        g = x;
        b = c;
    elif(hue < 300):
        r = x;
        g = 0;
        b = c;
    elif(hue < 360):
        r = c;
        g = 0;
        b = x;

    
    #hue = int(hue * 60 + 0.5); # °
    #sat = int(sat * 100 + 0.5); # %
    #lum = int(lum * 100 + 0.5); # %
    # to get paint values
    r = int((r + m) * 255 + 0.5); # °
    g = int((g + m) * 255 + 0.5); # %
    b = int((b + m) * 255 + 0.5); # %
    return [r, g, b];


def img_hsl2rgb(img_hsl):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation
    # convert hsl [0,255] range to [0,1]
    # When 0 ≤ H < 360, 0 ≤ S ≤ 1 and 0 ≤ L ≤ 1:
    img_hsl[:,:,0] = img_hsl[:,:,0] / 240.0 * 360.0
    img_hsl[:,:,1] = img_hsl[:,:,1] / 240.0 
    img_hsl[:,:,2] = img_hsl[:,:,2] / 240.0 

    #C = (1 - |2L - 1|) × S
    c = (1 - np.abs(2*img_hsl[:,:,2]-1)) * img_hsl[:,:,1]
    
    #X = C × (1 - |(H / 60°) mod 2 - 1|)
    x = c * (1 - np.abs((img_hsl[:,:,0] / 60) % 2 - 1))

    #m = L - C/2
    m = img_hsl[:,:,2] - c/2.0

    img_rgb = np.zeros(img_hsl.shape);
    img_r = np.zeros((img_hsl.shape[0], img_hsl.shape[1]));
    img_g = np.zeros((img_hsl.shape[0], img_hsl.shape[1]));
    img_b = np.zeros((img_hsl.shape[0], img_hsl.shape[1]));
    if(hue < 60):
        r = c;
        g = x;
        b = 0;
    elif(hue < 120):
        r = x;
        g = c;
        b = 0;
    elif(hue < 180):
        r = 0;
        g = c;
        b = x;
    elif(hue < 240):
        r = 0;
        g = x;
        b = c;
    elif(hue < 300):
        r = x;
        g = 0;
        b = c;
    elif(hue < 360):
        r = c;
        g = 0;
        b = x;

    
    #hue = int(hue * 60 + 0.5); # °
    #sat = int(sat * 100 + 0.5); # %
    #lum = int(lum * 100 + 0.5); # %
    # to get paint values
    r = int((r + m) * 255 + 0.5); # °
    g = int((g + m) * 255 + 0.5); # %
    b = int((b + m) * 255 + 0.5); # %
    return [r, g, b];

def preprocess_lum(img_rgb):
    img_rgb = img_rgb.astype(float);


    for y in range(img_rgb.shape[0]):
        print('y=',y)
        for x in range(img_rgb.shape[0]):
            pix_rgb = img_rgb[y,x];

            ## v0
            #if(pix_rgb[0] >= 190 and pix_rgb[1] >= 190 and pix_rgb[2] >= 190 and (np.abs(pix_rgb[0] - pix_rgb[1]) < 15 and np.abs(pix_rgb[0] - pix_rgb[2]) < 15 and np.abs(pix_rgb[1] - pix_rgb[2]) < 15)):
            #    continue;
            #pix_hsl = rgb2hsl(pix_rgb);
            #if(pix_hsl[2] > 90):
            #    pix_hsl[2]=80;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;
            
            ## v1
            #if((np.abs(pix_rgb[0] - pix_rgb[1]) < 15 and np.abs(pix_rgb[0] - pix_rgb[2]) < 15 and np.abs(pix_rgb[1] - pix_rgb[2]) < 15)):
            #    continue;
            #if(pix_rgb[0] >= 190 and pix_rgb[1] >= 190 and pix_rgb[2] >= 190 and (np.abs(pix_rgb[0] - pix_rgb[1]) < 15 and np.abs(pix_rgb[0] - pix_rgb[2]) < 15 and np.abs(pix_rgb[1] - pix_rgb[2]) < 15)):
            #    continue;
            #pix_hsl = rgb2hsl(pix_rgb);
            #if(pix_hsl[2] > 90):
            #    pix_hsl[2]=80;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;

            ##v2
            #if((np.abs(pix_rgb[0] - pix_rgb[1]) < 20 and np.abs(pix_rgb[0] - pix_rgb[2]) < 20 and np.abs(pix_rgb[1] - pix_rgb[2]) < 20)):
            #    continue;
            #if(pix_rgb[0] >= 190 and pix_rgb[1] >= 190 and pix_rgb[2] >= 190 and (np.abs(pix_rgb[0] - pix_rgb[1]) < 15 and np.abs(pix_rgb[0] - pix_rgb[2]) < 15 and np.abs(pix_rgb[1] - pix_rgb[2]) < 15)):
            #    continue;
            #pix_hsl = rgb2hsl(pix_rgb);
            #if(pix_hsl[2] > 90):
            #    pix_hsl[2]=80;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;

            ##v3
            #pix_hsl = rgb2hsl(pix_rgb);
            #pix_hsl[2]=80;
            #pix_rgb = hsl2rgb(pix_hsl);
            #img_rgb[y,x] = pix_rgb;

            #v4
            pix_hsl = rgb2hsl(pix_rgb);
            if(pix_hsl[2] > 80):
                pix_hsl[2]=80;
                pix_rgb = hsl2rgb(pix_hsl);
                img_rgb[y,x] = pix_rgb;

            ##v5
            #pix_hsl = rgb2hsl(pix_rgb);
            #if(pix_hsl[2] <= 80):
            #    continue;
            #elif(pix_hsl[2] <= 120):
            #    pix_hsl[2]=60;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;
            #elif(pix_hsl[2] <= 160):
            #    pix_hsl[2]=80;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;
            #elif(pix_hsl[2] <= 200):
            #    pix_hsl[2]=100;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;
            #elif(pix_hsl[2] <= 240):
            #    pix_hsl[2]=120;
            #    pix_rgb = hsl2rgb(pix_hsl);
            #    img_rgb[y,x] = pix_rgb;




    img = (img_rgb).astype(np.uint8)
    return img;
