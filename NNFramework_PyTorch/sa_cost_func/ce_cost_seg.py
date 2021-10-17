import sys;
import os;
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sa_net_cost_func import AbstractCostFunc;
from ..sa_net_loss_func_helper import CNNLossFuncHelper;

class CrossEntropyCostSeg:
    def __init__(self, n_classes, deviceID, kwargs):
        # predefined list of arguments
        args = {'class_weights':None};

        args.update(kwargs);
        class_weights = args['class_weights'];
        self.n_classes = n_classes;
        if(class_weights is not None):
            class_weights = torch.tensor([float(x) for x in class_weights.split(',')]).to(deviceID)
            self.cost_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.cost_fn = nn.CrossEntropyLoss()

        #if(class_weights == None):
        #    self.class_weights = tf.Variable(tf.ones([self.n_classes]));
        #else:
        #    self.class_weights = tf.Variable(class_weights);

        #self.cost_fn = nn.CrossEntropyLoss(reduction='sum')
        #self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        #self.nll_loss = nn.NLLLoss2d(class_weights)

    def calc_cost(self, logits, labels, deviceID):
        shape_diff_y = int((logits.shape[2] - labels.shape[1])/2)
        shape_diff_x = int((logits.shape[3] - labels.shape[2])/2)

        cost = self.cost_fn(logits[:,:,shape_diff_y:-shape_diff_y,shape_diff_x:-shape_diff_x], labels);
        #print('cost total = ', cost)
        #cost /= logits.shape[0]
        #print('cost avg = ', cost)
        #nll = self.nll_loss(F.log_softmax(logits[:,:,shape_diff_y:-shape_diff_y,shape_diff_x:-shape_diff_x]), labels)
        #print('nll_loss avg = ', nll)

        return cost;

    
