import sys;
import os;
#import tensorflow as tf;
import numpy as np;
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sa_net_cost_func import AbstractCostFunc;
from ..sa_net_loss_func_helper import CNNLossFuncHelper;
import math;
from distutils.util import strtobool;
from ..sa_helpers import multiplex_utils ;


class MSEOpticalDensityLossRGBDots:
    def __init__(self, n_classes, kwargs):
        # predefined list of arguments
        args = {'mse_all':'True', 'mse_all_type':'od', 'mse_all_scale_val':1.0, 'mse_all_weight':1.0
            , 'mse_maxonly':'False', 'mse_maxonly_type':'od-norm-1way', 'mse_maxonly_scale_val':1.0, 'mse_maxonly_weight':1.0
            , 'l1':'False', 'l1_lambda':0.0001, 'l1_maximize':'False', 'l1_maximize_lambda':0.0001
            , 'do_stain_diff': 'False'
        };
        #mse_all_type: [od,  od-norm-1way, rgb-norm-2way]
        #mse_maxonly_type: [od,  od-norm-1way, rgb-norm-2way]
        args.update(kwargs);
        print(args)

        self.cost_fn = nn.MSELoss();
        #self.cost_fn_sum = nn.MSELoss(reduction='sum');
        #self.cost_fn = self.my_mse_loss;
        #self.cost_fn = nn.BCELoss();


        self.one_hot = None;
        self.zero_const = None;
        self.one_const = None;

        self.do_mse_all = bool(strtobool(args['mse_all']));
        self.mse_all_type = str(args['mse_all_type'])
        self.mse_all_scale_val = float(args['mse_all_scale_val'])
        self.mse_all_weight = float(args['mse_all_weight'])

        self.do_mse_maxonly = bool(strtobool(args['mse_maxonly']));
        self.mse_maxonly_type = str(args['mse_maxonly_type'])
        self.mse_maxonly_scale_val = float(args['mse_maxonly_scale_val'])
        self.mse_maxonly_weight = float(args['mse_maxonly_weight'])

        self.do_L1 = bool(strtobool(args['l1']));
        self.do_L1_maximize = bool(strtobool(args['l1_maximize']));
        self.L1_lambda = float(args['l1_lambda']);
        self.L1_maximize_lambda = float(args['l1_maximize_lambda']);

        self.do_stain_diff = bool(strtobool(args['do_stain_diff']));

        print('self.do_mse_all = ', self.do_mse_all)
        print('self.mse_all_type = ', self.mse_all_type)
        print('self.mse_all_scale_val = ', self.mse_all_scale_val)

        print('self.do_mse_maxonly = ', self.do_mse_maxonly)
        print('self.mse_maxonly_type = ', self.mse_maxonly_type)
        print('self.mse_maxonly_scale_val = ', self.mse_maxonly_scale_val)
        
        print('self.do_L1 = ', self.do_L1)
        print('self.do_L1_maximize = ', self.do_L1_maximize)

        print('do_stain_diff = ', self.do_stain_diff)
    

    def calc_cost(self, logits, labels, concentration_logits, concentration_softmax, stains, deviceID):
        #print(labels.size());
        labels_od = multiplex_utils.transform_intensity_to_optical_density(labels[:,:,:,0:3]);
        labels_od = labels_od.permute(0,3,1, 2); 

        #print(logits.size());
        #print(labels_od.size());
        loss = self.cost_fn(logits, (labels_od)) *(255.0/np.log(255.0))*(255.0/np.log(255.0)) ;
        #loss = self.cost_fn(concentration_logits, (labels)) ;
        #loss = self.cost_fn(concentration_logits, (labels/255)) ;
        #loss = self.cost_fn(concentration_logits, (labels/255-0.5)*2) * logits.size()[0];

        labels_exist = labels[:,:,:,4:].sum(dim=-1);#####################################
        print('labels_exist.shape = ', labels_exist.shape)
        labels_conc = labels[:,:,:,4:].permute(0,3,1,2);
        print('labels_conc.shape = ', labels_conc.shape)
        print('concentration_logits.shape = ', concentration_logits.shape)
        #print('((concentration_logits - labels_conc)**2) : ', ((concentration_logits - labels_conc)**2).shape)
        labels_conc_flatten = labels_conc.reshape(labels_conc.shape[0], labels_conc.shape[1], labels_conc.shape[2]*labels_conc.shape[3])
        concentration_logits_flatten = concentration_logits.view(concentration_logits.size()[0], concentration_logits.size()[1], concentration_logits.size()[2] * concentration_logits.size()[3])
        img_label_od_flatten = torch.matmul(stains, (concentration_logits_flatten * labels_conc_flatten))
        img_label_od = img_label_od_flatten.view(-1, 3, concentration_logits.size()[2], concentration_logits.size()[3])
        #loss1 = (labels_exist * ((concentration_logits - labels_conc)**2).sum(dim=1)).sum()/labels_exist.sum()
        loss1 = self.cost_fn(img_label_od, (labels_od)) *(255.0/np.log(255.0))*(255.0/np.log(255.0)) ;
        #print('labels_conc = ', labels_conc);
        #print('labels_exist = ', labels_exist);
        print('loss0 = ', loss);
        print('loss1 = ', loss1);
        print('');
        loss += loss1

        return loss;



