#import tensorflow as tf;
import torch.optim as optim
import os;
from distutils.util import strtobool;
import numpy as np;
import torch
import glob;
import skimage.io as io;

#from ..sa_net_train import CNNTrainer;
#from sa_net_arch import AbstractCNNArch;
#from sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer_pytorch import OptimizerTypesPyTorch, CNNOptimizerPyTorch;
from ..sa_net_data_provider import AbstractDataProvider;
from ..sa_helpers import multiplex_utils;


class MultiplexAutoencoderTesterRGBExternalInput:
    def __init__(self, cnn_arch, cnn_arch_module, session_config, output_dir, device, output_ext, kwargs):
        # predefined list of arguments
        args = {'split_name':'test', 'batch_size':1, 'invert_out_img':'False', 'is_output_od':'False'};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        self.cnn_arch_module = cnn_arch_module;

        #self.test_data_provider = test_data_provider;
        self.batch_size = int(args['batch_size']);

        #self.invert_out_img = bool(strtobool(args['invert_out_img']));
        #self.is_output_od = bool(strtobool(args['is_output_od']));
        self.pad_y = int(args['pad_y']);
        self.pad_x = int(args['pad_x']);
        self.pad_y1 = int(np.floor(self.pad_y / 2.0));
        self.pad_y2 = int(np.ceil(self.pad_y / 2.0));
        self.pad_x1 = int(np.floor(self.pad_y / 2.0));
        self.pad_x2 = int(np.ceil(self.pad_y / 2.0));

        self.device = device;
        self.output_dir = output_dir;
        self.output_ext = output_ext;

    def init_model(self, do_init, do_restore):
        if(do_restore):
            checkpoint = self.cnn_arch_module.restore_model(is_test=True);            
            if(checkpoint is not None):
                print('restore succeed')

    def predict(self, inputs):
        with torch.no_grad():
            batch_x = inputs;
            if (batch_x is None):
                return None;
            
            batch_x = self.preprocess_input(inputs);

            if(self.device is not None):
                batch_x = batch_x.to(self.device);

            batch_y, batch_concentration, batch_concentration_sm, batch_stains = self.cnn_arch(batch_x);

            return batch_concentration.detach().cpu().numpy().astype(np.float16);


  
    def preprocess_input(self, inputs):
        np.clip(inputs, 0, 255, inputs);

        if(self.pad_y > 0 or self.pad_x > 0):
            inputs = np.pad(inputs, ((0,0),(0,0),(self.pad_y1, self.pad_y2),(self.pad_x1, self.pad_x2)),'constant', constant_values=128);

        inputs = torch.tensor(inputs, dtype = torch.float); # to avoid the error:  Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

        return inputs;