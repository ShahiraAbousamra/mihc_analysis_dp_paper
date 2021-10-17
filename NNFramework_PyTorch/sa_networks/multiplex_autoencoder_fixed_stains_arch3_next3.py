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
from collections.abc import Iterable;



class MultiplexAutoencoderFixedStainsArch3Next3(nn.Module):
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, model_restore_filename, cost_func:AbstractCostFunc, device, kwargs):
        super(MultiplexAutoencoderFixedStainsArch3Next3, self).__init__();
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.model_restore_filename = model_restore_filename;
        self.current_model_checkpoint_path = None;
        self.cost_func = cost_func;
        self.device = device;

        self.stains_dict = {};
        stains = np.array([[151, 82, 62], [118, 62, 151], [106, 106, 106], [62, 147, 151], [165, 168, 45], [174, 38, 75], [62, 104, 151], [221, 220, 219]]).transpose();
        self.stains_dict['shahira_wsi'] = stains;
        self.stains_dict['shahira_wsi-inv'] = 255 - stains;
        stains = np.array([[151, 82, 62], [118, 62, 151], [30, 30, 30], [62, 147, 151], [165, 168, 45], [174, 38, 75], [62, 104, 151], [221, 220, 219]]).transpose();
        self.stains_dict['shahira_wsi2'] = stains;
        self.stains_dict['shahira_wsi2-inv'] = 255 - stains;


        self.create_model(kwargs);



    def create_model(self, kwargs):
        # predefined list of arguments
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':'False', 'stain_init_name': 'dan-inv'
            , 'conv_init': 'uniform', 'use_softmax':'True', 'n_channels':3
            , 'dropout_keep_prob' : 1.0
            , 'device' : torch.device("cpu")
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
        self.n_channels = int(args['n_channels']);
        self.n_stains = self.stains_dict[self.stain_init_name].shape[1];
        #print('self.n_stains = ', self.n_stains)
        #print(self.stains_dict[self.stain_init_name])
        self.dropout_keep_prob = float(args['dropout_keep_prob'])

        print('self.pretrained = ', self.pretrained);

        #self.model = torchvision.models.vgg19(pretrained=self.pretrained);
        #if(self.input_img_width == 460 and self.input_img_height == 460):
        #    self.model.classifier[0] = nn.Linear(100352, 4096);
        ## modify the last layer
        #self.model.classifier[-1] = nn.Linear(4096, self.n_classes);

        # input is 3 channels optical density od = -np.log((img+1)/255.0);
        # number of output channels = (brown + 5 colors(black, yellow, teal, purple, red, [yellow+black, yellow+teal, yellow+purple]) + nucleus color ) + basic stains
        # = 1+5[+3]+1 = 7 or 10
        #self.n_stains = 8;
        #self.n_stains = 3;
        self.encoder = nn.Sequential(                    
            nn.Conv2d(self.n_channels, 128, kernel_size=1),  
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3), 
            nn.MaxPool2d(2, stride=2),                   
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True), 
            nn.Conv2d(64, 32, kernel_size=3),            
            nn.MaxPool2d(2, stride=2),                   
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3),           
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True));

        if(self.pretrained):
            self.decoder = nn.Sequential(             
                nn.ConvTranspose2d(16, 16, stride=2, kernel_size=3),  
                nn.Conv2d(16, 32, kernel_size=3),                     
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3),  
                nn.Conv2d(32, 64, kernel_size=3),                     
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3),                    
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.Conv2d(128, 3, kernel_size=1),         
                );
        else:
            self.decoder = nn.Sequential(             
                nn.ConvTranspose2d(16, 16, stride=2, kernel_size=3),  
                nn.Conv2d(16, 32, kernel_size=3),                     
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3),  
                nn.Conv2d(32, 64, kernel_size=3),                     
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3),                    
                nn.Dropout(p=self.dropout_keep_prob), 
                nn.ReLU(True),
                nn.Conv2d(128, self.n_stains, kernel_size=1),         
                );



        self.init_layers(self.encoder);
        self.init_layers(self.decoder);

        self.softmax_layer = torch.nn.Softmax(dim=1);

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

        init_stains_rgb = self.stains_dict[self.stain_init_name];

        #print('init_stains_rgb.shape = ', init_stains_rgb.shape);
        init_stains_od = self.transform_intensity_to_optical_density(init_stains_rgb);
        self.od_stains_tensor = torch.from_numpy(init_stains_od).float();
        #print('self.od_stains_tensor = ', self.od_stains_tensor);        
        self.od_stains = nn.Parameter(data=self.od_stains_tensor, requires_grad=False);
        print('self.od_stains = ', self.od_stains);
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

    def update_pretrained_model(self):

        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        self.decoder[-1] = nn.Conv2d(128, self.n_stains, kernel_size=1);
        self.decoder[-1].to(self.device);
        #print('after update_pretrained_model')
        #print(self.decoder[-2]);
        #print(self.decoder[-1]);


    def forward(self, x):
        #x = np.pad(x, ((0,0),(0,0),(10,10),(10,10)),'constant')
        #print('x.size = ', x.size());
        #print('self.od_stains.size = ', self.od_stains.size());
        x = self.encoder(x);
        #print('encoder.size = ', x.size());
        c = self.decoder(x);
        c1 = c**2; ############################### use the squares  to avoid negative values
        #c1 = torch.nn.functional.normalize(c, dim=1);
        #print('decoder.size = ', c.size());
        c2 = c1.view(-1, self.n_stains, c1.size()[2]*c1.size()[3]); # flatten the stains before multiply
        #print('c2.size = ', c2.size());
        #print('self.od_stains = ', self.od_stains);
        #print('c2 = ', c2);
        #if(self.use_softmax):
        #    sm = self.softmax_layer(c2);
        #else:
        #    sm = c2;
        #print(self.od_stains.shape)
        #print(c2.shape)
        o = torch.matmul(self.od_stains, c2);
        #print('o.size = ', o.size());
        #print('o.view(-1, 3, c.size()[2], c.size()[3]).size = ', o.view(-1, 3, c.size()[2], c.size()[3]).size());
        
        #return o.view(-1, 3, c.size()[2], c.size()[3]), c, sm.view(-1, self.n_stains, c.size()[2], c.size()[3]), self.od_stains;
        return o.view(-1, self.n_channels, c.size()[2], c.size()[3]), c1, c2.view(-1, self.n_stains, c.size()[2], c.size()[3]), self.od_stains;

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
        if(self.pretrained):
            print(self.checkpoint['model_state_dict'].keys());
            self.checkpoint['model_state_dict']['od_stains']=self.state_dict()['od_stains'] ;
            print(self.checkpoint['model_state_dict'].keys());
            #print(self.state_dict().keys());
            #self.state_dict().pop('od_stains', None)
            #print(self.state_dict().keys());
            self.load_state_dict(self.checkpoint['model_state_dict']);
            self.update_pretrained_model();
        else:
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

    def transform_intensity_to_optical_density(self, img_rgb):    
        od = -np.log((img_rgb+1)/255.0); 
        #print(od.shape);
        return od ;

    def init_layers(self, layer):    
        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
            if(self.conv_init == 'normal'):
                torch.nn.init.normal_(layer.weight) ;
            elif(self.conv_init == 'xavier_uniform'):
                torch.nn.init.xavier_uniform_(layer.weight) ;
            elif(self.conv_init == 'xavier_normal'):
                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
            elif(self.conv_init == 'he'):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        elif(isinstance(layer, Iterable)):
            for sub_layer in layer:
                self.init_layers(sub_layer);