import sys;
import os;
#import tensorflow as tf;
import numpy as np;

from ..sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
#from sa_net_arch import AbstractCNNArch;
from ..sa_net_cost_func import AbstractCostFunc;

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob;
import torchvision.models;
from distutils.util import strtobool;



class MultiplexSegUnet(nn.Module):
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, model_restore_filename, cost_func:AbstractCostFunc, kwargs):
        super(MultiplexSegUnet, self).__init__();
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.model_restore_filename = model_restore_filename;
        self.current_model_checkpoint_path = None;
        self.cost_func = cost_func;

        #self.stains_dict = {};
        #stains = np.array([[220, 168, 153], [223, 141, 228], [140, 123, 148], [107, 242, 248], [247, 234, 94], [239, 105, 159], [196, 215, 235], [248, 248, 247]]).transpose();
        #self.stains_dict['dan'] = stains;
        ##stains = np.array([[35, 87, 102], [32, 114, 27], [115, 132, 107], [148, 13, 7], [8, 21, 161], [16, 150, 96], [59, 40, 20], [7, 7, 8]]).transpose();
        ##self.stains_dict['dan-inv'] = stains;
        #self.stains_dict['dan-inv'] = 255 - stains;
        #stains = np.array([[145, 110, 90], [97, 19, 124], [33, 29, 34], [66, 126, 138], [194, 154, 52], [180, 46, 129], [97, 134, 175], [212, 212, 210]]).transpose();
        #self.stains_dict['shahira_dark'] = stains;
        #self.stains_dict['shahira_dark-inv'] = 255 - stains;
        #stains = np.array([[250, 5, 5], [5, 250, 5], [5, 5, 250]]).transpose();
        #self.stains_dict['rgb'] = stains;
        #self.stains_dict['rgb-inv'] = 255 - stains;

        self.create_model(kwargs);



    def create_model(self, kwargs):
        # predefined list of arguments
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':'False', 'stain_init_name': 'dan-inv'
            , 'conv_init': 'uniform'
            , 'use_softmax':'True', 'use_relu':'False', 'use_tanh':'False'
            ,'n_layers_per_path':3, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2
            , 'dropout_keep_prob' : 0.85
            , 'n_stains': 8
 
        };

        args.update(kwargs);

        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # read extra argument
        #self.n_layers_per_path = int(args['n_layers_per_path']); # n_layers_per_path in contracting path + n_layers_per_path in expanding path + 1 bottleneck layer
        #self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.input_img_width = int(args['input_img_width']);
        self.input_img_height = int(args['input_img_height']);
        #dropout = args['dropout'];
        self.pretrained = bool(strtobool(args['pretrained']));
        self.stain_init_name = str(args['stain_init_name']);
        self.conv_init = str(args['conv_init']).lower();
        self.use_softmax = bool(strtobool(args['use_softmax']));
        self.use_relu = bool(strtobool(args['use_relu']));
        self.use_tanh = bool(strtobool(args['use_tanh']));
    
        self.n_layers_per_path = int(args['n_layers_per_path']);
        self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.block_size = int(args['block_size']);
        self.pool_size = int(args['pool_size']);
        self.dropout_keep_prob = float(args['dropout_keep_prob'])

        self.n_stains = int(args['n_stains']);


        n_blocks = self.n_conv_blocks_in_start;
        n_blocks_prev = 3;

        # Contracting Path
        #self.encoder = [];
        self.encoder = nn.Sequential()
        layer_index = 0;
        for layer in range(0, self.n_layers_per_path):#264
    
            layer = nn.Sequential();
            if(layer_index > 0):
                layer.add_module('maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
            layer.add_module('conv_0_l_'+str(layer_index), nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
            #layer.add_module('dropout_0_l_'+str(layer_index), nn.Dropout(p=self.dropout_keep_prob));
            layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));
            layer.add_module('conv_1_l_'+str(layer_index), nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
            #layer.add_module('dropout_1_l_'+str(layer_index), nn.Dropout(p=self.dropout_keep_prob));
            layer.add_module('relu_1_l_'+str(layer_index), nn.ReLU(True));
            layer.add_module('dropout_0_l_'+str(layer_index), nn.Dropout(p=1-self.dropout_keep_prob));
            #self.encoder.append(layer);
            self.encoder.add_module('l_'+str(layer_index), layer);
            
            # updates for next level
            n_blocks_prev = n_blocks;
            n_blocks = n_blocks * 2;
            layer_index += 1

        # Bottleneck layer
        self.bottleneck = nn.Sequential();

        #self.bottleneck.add_module('dropout_0_l_bottleneck', nn.Dropout(p=self.dropout_keep_prob));
        self.bottleneck.add_module('maxpool_bottleneck', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        self.bottleneck.add_module('conv_0_bottleneck', nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
        #self.bottleneck.add_module('dropout_0_bottleneck', nn.Dropout(p=self.dropout_keep_prob));
        self.bottleneck.add_module('relu_0_bottleneck', nn.ReLU(True));
        self.bottleneck.add_module('conv_1_bottleneck', nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
        #self.bottleneck.add_module('dropout_1_bottleneck', nn.Dropout(p=self.dropout_keep_prob));
        self.bottleneck.add_module('relu_1_bottleneck', nn.ReLU(True));

        n_blocks_prev = n_blocks;
        n_blocks = n_blocks // 2;

        # Expanding Path
        #self.decoder = [];
        self.decoder = nn.Sequential()
        layer_index = 0;
        for layer in range(0, self.n_layers_per_path):
            layer = nn.Sequential();
            layer.add_module('deconv_l_'+str(layer_index), nn.ConvTranspose2d(n_blocks_prev, n_blocks_prev//2, stride=self.pool_size, kernel_size=self.pool_size))
            layer.add_module('conv_0_l_'+str(layer_index), nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
            #layer.add_module('dropout_0_l_'+str(layer_index), nn.Dropout(p=self.dropout_keep_prob));
            layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));
            layer.add_module('conv_1_l_'+str(layer_index), nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
            #layer.add_module('dropout_1_l_'+str(layer_index), nn.Dropout(p=self.dropout_keep_prob));
            layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));

            self.decoder.add_module('l_'+str(layer_index), layer);


  
            # updates for next level
            n_blocks_prev = n_blocks;
            n_blocks = n_blocks // 2;   
            layer_index += 1


        # Final Layer
        self.final_layer = nn.Sequential();
        self.final_layer.add_module('conv_final', nn.Conv2d(n_blocks_prev, self.n_stains, kernel_size=1));
        if(self.use_relu):
            self.final_layer.add_module('relu_final', nn.ReLU(True));
        if(self.use_tanh):
            self.final_layer.add_module('tanh_final', nn.Tanh());
    
        # Softmax
        self.softmax_layer = torch.nn.Softmax(dim=1);

        #self.model = torchvision.models.vgg19(pretrained=self.pretrained);
        #if(self.input_img_width == 460 and self.input_img_height == 460):
        #    self.model.classifier[0] = nn.Linear(100352, 4096);
        ## modify the last layer
        #self.model.classifier[-1] = nn.Linear(4096, self.n_classes);

        # input is 3 channels optical density od = -np.log((img+1)/255.0);
        # number of output channels = (brown + 5 colors(black, yellow, teal, purple, red, [yellow+black, yellow+teal, yellow+purple]) + nucleus color ) + basic stains
        # = 1+5[+3]+1 = 7 or 10
        #self.n_stains = 3;


        BIAS_INIT = 0.1;
        for l in self.encoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                        #layer.bias.data.fill_(BIAS_INIT);

        for layer in self.bottleneck:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                    #layer.bias.data.fill_(BIAS_INIT);

        for l in self.decoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                        #layer.bias.data.fill_(BIAS_INIT);

        for layer in self.final_layer:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                    #layer.bias.data.fill_(BIAS_INIT);

        #self.od_stains_tensor = torch.rand(3, self.n_stains);
        
        # order of stains is: 
        #  brown:  35 87 102 -->  (255 - X) --> 220  168  153
        #  purple: 32 114 27 -->  (255 - X) --> 223  141  228
        #  black/silver:  115 132 107 -->  (255 - X) --> 140  123  148
        #  teal: 148 13 7 -->  (255 - X) --> 107 242  248
        #  yellow:  8 21 161  -->  (255 - X) --> 247 234  94
        #  red: 16 150 96 -->  (255 - X) --> 239  105	159
        #  blue: 59 40 20  -->  (255 - X) -->  196 215  235
        #  background: 7 7 8  -->  (255 - X) -->  248 248 247

        #init_stains_rgb = self.stains_dict[self.stain_init_name];

        ##print('init_stains_rgb.shape = ', init_stains_rgb.shape);
        #init_stains_od = self.transform_intensity_to_optical_density(init_stains_rgb);
        #self.od_stains_tensor = torch.from_numpy(init_stains_od).float();
        ##print('self.od_stains_tensor = ', self.od_stains_tensor);        
        #self.od_stains = nn.Parameter(data=self.od_stains_tensor, requires_grad=False);
        #print('self.od_stains = ', self.od_stains);

        #self.od_reconstruct = nn.Sequential(             
        #    nn.ConvTranspose2d(7,1,kernel_size=1),
        #    nn.ReLU(True));

        ## print the model layers:
        #child_counter = 0
        #for child in self.model.children():
        #   print(" child", child_counter, "is:")
        #   print(child)
        #   child_counter += 1

        # Get all layers except last
        #self.L1 = nn.Sequential(*list(self.model.children())[:-1]);
        #self.FC = nn.Linear(512, self.n_classes);

        # modify the last layer
        #self.model.fc = nn.Linear(73728, self.n_classes);

        self.zero_grad() ;
        #out.backward(torch.randn(1, self.n_classes)) ;

        #return output_map, variables; ####################################

    #def create_model(self, kwargs):
    #    # predefined list of arguments
    #    args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':'False', 'stain_init_name': 'dan-inv'
    #        , 'conv_init': 'uniform', 'use_softmax':'True'
    #        ,'n_layers_per_path':3, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2
    #        , 'n_stains': 8
 
    #    };

    #    args.update(kwargs);

    #    # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

    #    # read extra argument
    #    #self.n_layers_per_path = int(args['n_layers_per_path']); # n_layers_per_path in contracting path + n_layers_per_path in expanding path + 1 bottleneck layer
    #    #self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
    #    self.input_img_width = int(args['input_img_width']);
    #    self.input_img_height = int(args['input_img_height']);
    #    #dropout = args['dropout'];
    #    self.pretrained = bool(strtobool(args['pretrained']));
    #    self.stain_init_name = str(args['stain_init_name']);
    #    self.conv_init = str(args['conv_init']).lower();
    #    self.use_softmax = bool(strtobool(args['use_softmax']));
    
    #    self.n_layers_per_path = int(args['n_layers_per_path']);
    #    self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
    #    self.block_size = int(args['block_size']);
    #    self.pool_size = int(args['pool_size']);

    #    self.n_stains = int(args['n_stains']);


    #    n_blocks = self.n_conv_blocks_in_start;
    #    n_blocks_prev = 3;

    #    # Contracting Path
    #    #self.encoder = [];
    #    self.encoder = nn.Sequential()
    #    layer_index = 0;
    #    for layer in range(0, self.n_layers_per_path):#264
    
    #        layer = nn.Sequential();
    #        if(layer_index > 0):
    #            layer.add_module('maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
    #        layer.add_module('conv_0_l_'+str(layer_index), nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
    #        layer.add_module('dropout_0_l_'+str(layer_index), nn.Dropout(p=0.85));
    #        layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));
    #        layer.add_module('conv_1_l_'+str(layer_index), nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
    #        layer.add_module('dropout_1_l_'+str(layer_index), nn.Dropout(p=0.85));
    #        layer.add_module('relu_1_l_'+str(layer_index), nn.ReLU(True));
    #        #self.encoder.append(layer);
    #        self.encoder.add_module('l_'+str(layer_index), layer);
            
    #        # updates for next level
    #        n_blocks_prev = n_blocks;
    #        n_blocks = n_blocks * 2;
    #        layer_index += 1

    #    # Bottleneck layer
    #    self.bottleneck = nn.Sequential();

    #    self.bottleneck.add_module('maxpool_bottleneck', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
    #    self.bottleneck.add_module('conv_0_bottleneck', nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
    #    self.bottleneck.add_module('dropout_0_bottleneck', nn.Dropout(p=0.85));
    #    self.bottleneck.add_module('relu_0_bottleneck', nn.ReLU(True));
    #    self.bottleneck.add_module('conv_1_bottleneck', nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
    #    self.bottleneck.add_module('dropout_1_bottleneck', nn.Dropout(p=0.85));
    #    self.bottleneck.add_module('relu_1_bottleneck', nn.ReLU(True));
    #    #self.bottleneck.add_module('maxpool_bottleneck', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));

    #    n_blocks_prev = n_blocks;
    #    n_blocks = n_blocks // 2;

    #    # Expanding Path
    #    #self.decoder = [];
    #    self.decoder = nn.Sequential()
    #    layer_index = 0;
    #    for layer in range(0, self.n_layers_per_path):
    #        layer = nn.Sequential();
    #        #layer.add_module('deconv_l_'+str(layer_index), nn.ConvTranspose2d(n_blocks_prev, n_blocks_prev//2, stride=self.pool_size, kernel_size=self.pool_size))
    #        layer.add_module('deconv_l_'+str(layer_index), nn.ConvTranspose2d(n_blocks_prev, n_blocks_prev, stride=self.pool_size, kernel_size=self.pool_size))
    #        layer.add_module('conv_0_l_'+str(layer_index), nn.Conv2d(n_blocks_prev, n_blocks, kernel_size=self.block_size));
    #        layer.add_module('dropout_0_l_'+str(layer_index), nn.Dropout(p=0.85));
    #        layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));
    #        layer.add_module('conv_1_l_'+str(layer_index), nn.Conv2d(n_blocks, n_blocks, kernel_size=self.block_size));
    #        layer.add_module('dropout_1_l_'+str(layer_index), nn.Dropout(p=0.85));
    #        layer.add_module('relu_0_l_'+str(layer_index), nn.ReLU(True));

    #        self.decoder.add_module('l_'+str(layer_index), layer);

    #        ## get corresponding layer in contracting path
    #        #corresponding_layer_indx = n_layers_per_path - layer -1;
    #        ## crop and concatenate
    #        #cropped = CNNArchUtils.crop_a_to_b(conv_out[corresponding_layer_indx][1],  up_pool);
    #        #concatenated = CNNArchUtils.concate_a_to_b(cropped, up_pool);

  
    #        # updates for next level
    #        n_blocks_prev = n_blocks;
    #        n_blocks = n_blocks // 2;   
    #        layer_index += 1

    #    # Final Layer
    #    self.final_layer = nn.Sequential();
    #    self.final_layer.add_module('conv_final', nn.Conv2d(n_blocks_prev, self.n_stains, kernel_size=1));

    #    # Softmax
    #    self.softmax_layer = torch.nn.Softmax(dim=1);

    #    #self.model = torchvision.models.vgg19(pretrained=self.pretrained);
    #    #if(self.input_img_width == 460 and self.input_img_height == 460):
    #    #    self.model.classifier[0] = nn.Linear(100352, 4096);
    #    ## modify the last layer
    #    #self.model.classifier[-1] = nn.Linear(4096, self.n_classes);

    #    # input is 3 channels optical density od = -np.log((img+1)/255.0);
    #    # number of output channels = (brown + 5 colors(black, yellow, teal, purple, red, [yellow+black, yellow+teal, yellow+purple]) + nucleus color ) + basic stains
    #    # = 1+5[+3]+1 = 7 or 10
    #    #self.n_stains = 3;


    #    BIAS_INIT = 0.1;
    #    for l in self.encoder:
    #        for layer in l:
    #            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
    #                if(self.conv_init == 'normal'):
    #                    torch.nn.init.normal_(layer.weight) ;
    #                elif(self.conv_init == 'xavier_uniform'):
    #                    torch.nn.init.xavier_uniform_(layer.weight) ;
    #                elif(self.conv_init == 'xavier_normal'):
    #                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
    #                elif(self.conv_init == 'he'):
    #                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
    #                    layer.bias.data.fill_(BIAS_INIT);

    #    for layer in self.bottleneck:
    #        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
    #            if(self.conv_init == 'normal'):
    #                torch.nn.init.normal_(layer.weight) ;
    #            elif(self.conv_init == 'xavier_uniform'):
    #                torch.nn.init.xavier_uniform_(layer.weight) ;
    #            elif(self.conv_init == 'xavier_normal'):
    #                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
    #            elif(self.conv_init == 'he'):
    #                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
    #                layer.bias.data.fill_(BIAS_INIT);

    #    for l in self.decoder:
    #        for layer in l:
    #            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
    #                if(self.conv_init == 'normal'):
    #                    torch.nn.init.normal_(layer.weight) ;
    #                elif(self.conv_init == 'xavier_uniform'):
    #                    torch.nn.init.xavier_uniform_(layer.weight) ;
    #                elif(self.conv_init == 'xavier_normal'):
    #                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
    #                elif(self.conv_init == 'he'):
    #                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
    #                    layer.bias.data.fill_(BIAS_INIT);

    #    for layer in self.final_layer:
    #        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
    #            if(self.conv_init == 'normal'):
    #                torch.nn.init.normal_(layer.weight) ;
    #            elif(self.conv_init == 'xavier_uniform'):
    #                torch.nn.init.xavier_uniform_(layer.weight) ;
    #            elif(self.conv_init == 'xavier_normal'):
    #                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
    #            elif(self.conv_init == 'he'):
    #                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
    #                layer.bias.data.fill_(BIAS_INIT);

    #    #self.od_stains_tensor = torch.rand(3, self.n_stains);
        
    #    # order of stains is: 
    #    #  brown:  35 87 102 -->  (255 - X) --> 220  168  153
    #    #  purple: 32 114 27 -->  (255 - X) --> 223  141  228
    #    #  black/silver:  115 132 107 -->  (255 - X) --> 140  123  148
    #    #  teal: 148 13 7 -->  (255 - X) --> 107 242  248
    #    #  yellow:  8 21 161  -->  (255 - X) --> 247 234  94
    #    #  red: 16 150 96 -->  (255 - X) --> 239  105	159
    #    #  blue: 59 40 20  -->  (255 - X) -->  196 215  235
    #    #  background: 7 7 8  -->  (255 - X) -->  248 248 247

    #    init_stains_rgb = self.stains_dict[self.stain_init_name];

    #    #print('init_stains_rgb.shape = ', init_stains_rgb.shape);
    #    init_stains_od = self.transform_intensity_to_optical_density(init_stains_rgb);
    #    self.od_stains_tensor = torch.from_numpy(init_stains_od).float();
    #    #print('self.od_stains_tensor = ', self.od_stains_tensor);        
    #    self.od_stains = nn.Parameter(data=self.od_stains_tensor, requires_grad=False);
    #    print('self.od_stains = ', self.od_stains);
    #    #self.od_reconstruct = nn.Sequential(             
    #    #    nn.ConvTranspose2d(7,1,kernel_size=1),
    #    #    nn.ReLU(True));

    #    ## print the model layers:
    #    #child_counter = 0
    #    #for child in self.model.children():
    #    #   print(" child", child_counter, "is:")
    #    #   print(child)
    #    #   child_counter += 1

    #    # Get all layers except last
    #    #self.L1 = nn.Sequential(*list(self.model.children())[:-1]);
    #    #self.FC = nn.Linear(512, self.n_classes);

    #    # modify the last layer
    #    #self.model.fc = nn.Linear(73728, self.n_classes);

    #    self.zero_grad() ;


    def forward(self, x):
        #print('np.unique(in)=',torch.unique(x))
        #print('in.size = ', x.size());
        #print('self.od_stains.size = ', self.od_stains.size());
        encoder_out = [];
        #print(self.encoder[0]);
        #print('input  x.size() = ', x.size())
        for l in self.encoder:     
            x = l(x);
            #print('after encoder layer x.size() = ', x.size())
            encoder_out.append(x);
        #print('np.unique(enc)=',torch.unique(x))
        #print('enc.shape',x.shape)
        x = self.bottleneck(x);
        #print('np.unique(bottle)=',torch.unique(x))
        print('len np.unique(bottle)=',torch.unique(x).size())
        #print('bottle.shape',x.shape)
        j = self.n_layers_per_path;
        for l in self.decoder:            
            #print('before deconv x.size() = ', x.size())
            x = l[0](x);
            #print('after deconv x.size() = ', x.size())
            j -= 1;
            corresponding_layer_indx = j;
            ## crop and concatenate
            #print('before concat x.size() = ', x.size())
            #print('concat to size = ', encoder_out[corresponding_layer_indx].size());
            cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx],  x);
            x = torch.cat((cropped, x), 1) ;
            #print('after concat x.size() = ', x.size())
            for i in range(1, len(l)):
                x = l[i](x);
                #print('x.size() = ', x.size())

        #print('np.unique(dec)=',torch.unique(x))
        c = self.final_layer(x);
        #print('np.unique(final)=',torch.unique(c))

        #print('decoder.size = ', c.size());
        #c2 = c.view(-1, self.n_stains, c.size()[2]*c.size()[3]); # flatten the stains before multiply
        #print('c2.size = ', c2.size());
        #print('self.od_stains = ', self.od_stains);
        #print('c2 = ', c2);
        if(self.use_softmax):
            sm = self.softmax_layer(c);
        else:
            sm = c;
        #print('len np.unique(sm)=',torch.unique(sm).size())
        #o = torch.matmul(self.od_stains, sm);
        #o = torch.matmul(self.od_stains, c2);
        #print('sm.size = ', sm.size());
        #print('o.view(-1, 3, c.size()[2], c.size()[3]).size = ', o.view(-1, 3, c.size()[2], c.size()[3]).size());
        
        #return o.view(-1, 3, c.size()[2], c.size()[3]), c, self.od_stains;
        #return o.view(-1, 3, c.size()[2], c.size()[3]), c, sm.view(-1, self.n_stains, c.size()[2], c.size()[3]), self.od_stains;
        return sm;

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

 

    def get_prediction_softmax(self, logits):
        return CNNArchUtils.get_probability_softmax(logits);

    def get_class_prediction(self, logits):
        ## debug
        #print('logits={}'.format(logits));
        #print('logits.data={}'.format(logits.data));
        probability, predicted_class = torch.max(logits.data, 1)
        ## debug
        #print(predicted_class)
        return predicted_class ;

    def get_correct_prediction(self, logits, labels):
        prediction = self.get_class_prediction(logits);
        return (torch.eq(prediction, labels));


    def calc_out_size_conv2d(self, in_width, in_height, kernel_size):
        out_width = -1;
        out_height = -1;
        if(in_width > 0):
            out_width = in_width - kernel_size + 1;
        if(in_height > 0):
            out_height = in_height - kernel_size + 1;
        return out_width, out_height;

    def calc_out_size_maxpool2d(self, in_width, in_height, kernel_size):
        out_width = -1;
        out_height = -1;
        if(in_width > 0):
            out_width = in_width//2;
        if(in_height > 0):
            out_height = in_height//2;
        return out_width, out_height;


    def print_model_params(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def save_model(self, sess, optimizer, epoch, suffix=""):
        postfix = '_epoch_{:04d}'.format(epoch)+ suffix;
        self.filepath = os.path.join(self.model_out_path, self.model_base_filename+ postfix + '.pth');
        print('self.filepath = ', self.filepath);
        #torch.save(model.state_dict(), filepath);
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            #'loss': loss,
            }, self.filepath);
        return self.filepath;

    def restore_model(self, sess=None, is_test=True):
        print('inside restore');
        if(self.model_restore_filename is None):
            self.filepath = None;
            ##debug
            #print('self.model_restore_filename is None')
            return None;
        if(os.path.isfile(self.model_restore_filename)):
            self.filepath = self.model_restore_filename;
        elif(os.path.isfile(self.model_restore_filename + '.pth')):
            self.filepath = self.model_restore_filename + '.pth';
        else:
            self.filepath = os.path.join(self.model_out_path, self.model_restore_filename + '.pth');
        ##debug
        print('filepath =', self.filepath )
        if(not os.path.isfile(self.filepath)):
            filepath_pattern = os.path.join(self.model_out_path, self.model_base_filename + '*.pth');
            list_of_files = glob.glob(filepath_pattern);
            if(len(list_of_files) <= 0):
                return None;
            self.filepath = max(list_of_files);
            print(self.filepath);
            if(not os.path.isfile(self.filepath)):
                return None;
        #self.load_state_dict(torch.load(filepath))
        self.checkpoint = torch.load(self.filepath);
        self.load_state_dict(self.checkpoint['model_state_dict']);
        if(is_test):
            self.eval();
        else:
            self.train()
        return self.checkpoint;

    def change_mode(self, is_test=True):
        if(is_test):
            self.eval();
        else:
            self.train()

    #def transform_intensity_to_optical_density(self, img_rgb):    
    #    od = -np.log((img_rgb+1)/255.0); 
    #    #print(od.shape);
    #    return od ;

