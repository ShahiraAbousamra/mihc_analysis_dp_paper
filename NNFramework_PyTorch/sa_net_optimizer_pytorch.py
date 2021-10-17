#import tensorflow as tf;
from enum import Enum;
import torch.optim as optim;

class OptimizerTypesPyTorch(Enum):
    ADAM = 1,
    SGD = 2


class CNNOptimizerPyTorch:

    @staticmethod
    def adam_optimizer(learning_rate, cnn_arch, kwargs):
        args = {'weight_decay':0.0};
        args.update(kwargs);
        weight_decay = float(args['weight_decay']);
        print('weight_decay=',weight_decay);
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, cnn_arch.parameters()), lr=learning_rate, weight_decay=weight_decay);
        optimizer = optim.Adam(cnn_arch.parameters(), lr=learning_rate, weight_decay=weight_decay);
        optimizer.zero_grad();
        return optimizer;

    @staticmethod
    def sgd_optimizer(learning_rate, cnn_arch, kwargs):
        args = {'momentum':0.9};
        args.update(kwargs);
        momentum = float(args['momentum']);
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, cnn_arch.parameters()), lr=learning_rate, momentum=momentum);
        optimizer.zero_grad();
        return optimizer;

