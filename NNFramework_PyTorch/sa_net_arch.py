import sys;
import os;
import tensorflow as tf;
import numpy as np;

from sa_net_arch_utilities import CNNArchUtils;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
from sa_net_cost_func import AbstractCostFunc;

class AbstractCNNArch:
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, cost_func:AbstractCostFunc, kwargs):
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.current_model_checkpoint_path = None;
        self.input_x = tf.placeholder(tf.float32, [None, None, None, n_channels])
        self.labels = tf.placeholder(tf.float32, [None, None, None, n_classes])
        self.isTest = tf.placeholder(tf.bool);
        self.dropout = tf.placeholder(tf.float32);
        self.cost_func = cost_func;
        self.epochs_count = tf.Variable(0);
        self.epoch = tf.placeholder(tf.bool);
        self.dropout = tf.placeholder(tf.float32);

        self.logits, self.variables = self.create_model(self.input_x, self.isTest, self.dropout, kwargs);
        self.cost = self.cost_func.calc_cost(self.logits, self.labels);
        self.prediction_softmax = self.get_prediction_softmax(self.logits);
        self.prediction_class = self.get_class_prediction(self.logits);
        self.correct_pred = self.get_correct_prediction(self.logits, self.labels);
        self.accuracy = self.get_accuracy();
        self.saver = tf.train.Saver(max_to_keep=100000);

    def save_model(self, sess, postfix):
        filename = os.path.join(self.model_out_path, self.model_base_filename+ postfix + '.ckpt');
        self.saver.save(sess, filename );

    def restore_model(self, sess):
        #if(is_model_exist()):
        #    self.saver.restore(sess, self.model_out_path);
        #    return True;
        #return False;
        ckpt = tf.train.get_checkpoint_state(self.model_out_path);
        if ckpt and ckpt.model_checkpoint_path:
            self.current_model_checkpoint_path = ckpt.model_checkpoint_path;
            self.saver.restore(sess, ckpt.model_checkpoint_path);

    #def is_model_exist(self):
    #    if(os.path.isfile(self.model_out_path + '.ckpt.meta')):
    #        return True;
    #    return False;

    def create_model(self, input_x, isTest, dropout, kwargs):
        pass;
        ## predefined list of arguments
        #args = {'n_layers_per_path':3, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2};

        ### remove check to allow arguments that will not be used
        ### check arguments not in predefined list and show error if an unlisted argument is supplied
        ##diff = set(kwargs.keys()) - set(args.keys());
        ##if diff:
        ##    err_msg = "Invalid args to create_model: " + str(tuple(diff));
        ##    print(err_msg, file=sys.stderr);
        ##    raise ValueError(err_msg);
        #args.update(kwargs);

        ## read extra argument
        #n_layers_per_path = args['n_layers_per_path']; # n_layers_per_path in contracting path + n_layers_per_path in expanding path + 1 bottleneck layer
        #n_conv_blocks_in_start = args['n_conv_blocks_in_start'];
        #block_size = args['block_size'];
        #pool_size = args['pool_size'];
        ##dropout = args['dropout'];

        #weights = [];
        #biases = [];
        #conv_out= [];
        #down_out= [];
        #up_out= [];
        #input = tf.reshape(input_x, [-1, tf.shape(input_x)[1], tf.shape(input_x)[2], self.n_channels]);
        #n_blocks = n_conv_blocks_in_start;
        #n_blocks_prev = self.n_channels;

        ## Contracting Path
        #for layer in range(0, n_layers_per_path):
        #    '''
        #    Ideally the initial weights should be adapted such that each feature map in the network has approximately unit variance. 
        #    For a network with our architecture (alternating convolution and ReLU layers) 
        #    this can be achieved by drawing the initial weights from:
        #        a Gaussian distribution with a standard deviation of sqrt(2/N), 
        #        where N denotes the number of incoming nodes of one neuron [5]. 
        #    E.g. for a 3x3 convolution and 64 feature channels in the previous layer         
        #        N = 9 · 64 = 576.
        #    '''
        #    stddev = np.sqrt(2 / (block_size * block_size * n_blocks))
        #    w_a = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks_prev, n_blocks], stddev);
        #    w_b = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks, n_blocks], stddev);
        #    b_a = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);
        #    b_b = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);

        #    conv_a = CNNArchUtils.conv2d(input, w_a, b_a, stride=1);
        #    conv_a = CNNArchUtils.dropout(conv_a, dropout, isTest); # in u-net paper it is not clear where to apply dropout
        #    conv_a = CNNArchUtils.relu(conv_a);
        #    conv_b = CNNArchUtils.conv2d(conv_a, w_b, b_b, stride=1);
        #    conv_b = CNNArchUtils.dropout(conv_b, dropout, isTest);
        #    conv_b = CNNArchUtils.relu(conv_b);

        #    max_pool = CNNArchUtils.max_pool_2d(conv_b, k=pool_size, stride=pool_size);
            
        #    weights.append((w_a, w_b));
        #    biases.append((b_a, b_b));
        #    conv_out.append((conv_a, conv_b));
        #    down_out.append((max_pool));
            
        #    # updates for next level
        #    input = max_pool;
        #    n_blocks_prev = n_blocks;
        #    n_blocks = n_blocks * 2;
            
        ## Bottleneck layer

        #stddev = np.sqrt(2 / (block_size * block_size * n_blocks))
        #w_a = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks_prev, n_blocks], stddev);
        #w_b = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks, n_blocks], stddev);
        #b_a = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);
        #b_b = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);

        #conv_a = CNNArchUtils.conv2d(input, w_a, b_a, stride=1);
        #conv_a = CNNArchUtils.dropout(conv_a, dropout, isTest); # in u-net paper it is not clear where to apply dropout
        #conv_a = CNNArchUtils.relu(conv_a);

        #conv_b = CNNArchUtils.conv2d(conv_a, w_b, b_b, stride=1);
        #conv_b = CNNArchUtils.dropout(conv_b, dropout, isTest);
        #conv_b = CNNArchUtils.relu(conv_b);
  
        #weights.append((w_a, w_b));
        #biases.append((b_a, b_b));
        #conv_out.append((conv_a, conv_b));
        
        #input = conv_b;
        #n_blocks_prev = n_blocks;
        #n_blocks = n_blocks // 2;

        ## Expanding Path
        #for layer in range(0, n_layers_per_path):
        #    stddev = np.sqrt(2 / (block_size * block_size * n_blocks))
        #    w_deconv = CNNArchUtils.init_weights_normal([pool_size, pool_size, n_blocks_prev, n_blocks], stddev);
        #    up_pool = CNNArchUtils.deconv2d(input, w_deconv, k=pool_size, stride=pool_size);
        #    '''
        #    Ideally the initial weights should be adapted such that each feature map in the network has approximately unit variance. 
        #    For a network with our architecture (alternating convolution and ReLU layers) 
        #    this can be achieved by drawing the initial weights from:
        #        a Gaussian distribution with a standard deviation of sqrt(2/N), 
        #        where N denotes the number of incoming nodes of one neuron [5]. 
        #    E.g. for a 3x3 convolution and 64 feature channels in the previous layer         
        #        N = 9 · 64 = 576.
        #    '''
        #    #stddev = np.sqrt(2 / (block_size * block_size * n_blocks))
        #    w_a = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks_prev, n_blocks], stddev);
        #    w_b = CNNArchUtils.init_weights_normal([block_size, block_size, n_blocks, n_blocks], stddev);
        #    b_a = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);
        #    b_b = CNNArchUtils.init_weights_constant([n_blocks], const_val=0.1);

        #    # get corresponding layer in contracting path
        #    corresponding_layer_indx = n_layers_per_path - layer -1;
        #    # crop and concatenate
        #    cropped = CNNArchUtils.crop_a_to_b(conv_out[corresponding_layer_indx][1],  up_pool);
        #    concatenated = CNNArchUtils.concate_a_to_b(cropped, up_pool);

        #    conv_a = CNNArchUtils.conv2d(concatenated, w_a, b_a, stride=1);
        #    ## Drop-out layers at the end of the contracting path perform further implicit data augmentation
        #    ## to do: try drop out only after the bottleneck layer 
        #    conv_a = CNNArchUtils.dropout(conv_a, dropout, isTest); # in u-net paper it is not clear where to apply dropout
        #    conv_a = CNNArchUtils.relu(conv_a);

        #    conv_b = CNNArchUtils.conv2d(conv_a, w_b, b_b, stride=1);
        #    conv_b = CNNArchUtils.dropout(conv_b, dropout, isTest);
        #    conv_b = CNNArchUtils.relu(conv_b);

            
        #    weights.append((w_a, w_b));
        #    biases.append((b_a, b_b));
        #    up_out.append((up_pool));
        #    conv_out.append((conv_a, conv_b));
            
        #    # updates for next level
        #    input = conv_b;
        #    n_blocks_prev = n_blocks;
        #    n_blocks = n_blocks // 2;   

        ## final layer
        #w_final = CNNArchUtils.init_weights_normal([1, 1, n_blocks_prev, self.n_classes], stddev)
        #b_final = CNNArchUtils.init_weights_constant([self.n_classes], const_val=0.1); 
        #conv_final = CNNArchUtils.conv2d(input, w_final, b_final, stride=1);
        #conv_final = CNNArchUtils.relu(conv_final);
        #output_map = conv_final;

        #variables = []
        #for w1,w2 in weights:
        #    variables.append(w1)
        #    variables.append(w2)
        
        #for b1,b2 in biases:
        #    variables.append(b1)
        #    variables.append(b2)
                
        #return output_map, variables;

    #def calc_cost(self, logits, labels, kwargs):
    #    # predefined list of arguments
    #    args = {'cost_func': CostFuncTypes.CROSS_ENTROPY};
    #    args.update(kwargs);
    #    cost_type = args['cost_func'];

    #    if(cost_type == CostFuncTypes.CROSS_ENTROPY):
    #        return CNNLossFuncHelper.cost_cross_entropy(logits, labels, self.get_class_weights(logits, kwargs), self.n_classes);

    #    return 0;

    #def get_class_weights(self, logits, kwargs):
    #    ## predefined list of arguments
    #    #args = {'cost_func':'cross_entropy'};
    #    return self.class_weights;

    #def set_class_weights(self, weights, kwargs):
    #    ## predefined list of arguments
    #    #args = {'cost_func':'cross_entropy'};
    #    return self.class_weights.assign(weights);

    def get_prediction_softmax(self, logits):
        return CNNArchUtils.get_probability_softmax(logits);

    def get_class_prediction(self, logits):
        return tf.argmax(logits, axis = len(tf.shape(logits))-1);

    def get_correct_prediction(self, logits, labels):
        prediction = tf.argmax(logits, axis = len(tf.shape(logits))-1);
        label = tf.argmax(labels, axis = len(tf.shape(labels))-1);
        return tf.equal(prediction, label);

    def get_accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    #def get_prediction_size(self, sess, data_point):
    #    data_point = data_point.reshape(1, data_point.shape[0], data_point.shape[1], data_point.shape[2]);
    #    #print('data_point');
    #    #print(data_point.shape);
    #    out = sess.run([self.logits], feed_dict={self.input_x:data_point, self.isTest:True, self.dropout:0.0});
    #    #print(out);
    #    return np.shape(out);


    def get_prediction_size(self, sess, data_point):
        data_point = data_point.reshape(1, data_point.shape[0], data_point.shape[1], data_point.shape[2]);
        #tf.Print(data_point,[data_point]);
        #print(data_point.shape);
        out = sess.run([self.logits], feed_dict={self.input_x:data_point, self.isTest:True, self.dropout:0.0});
        #print('out');
        #print(np.shape(out));
        #tf.Print(out,[out]);
        shape = np.shape(out)
        out = np.reshape(out, (shape[1], shape[2], shape[3], shape[4]));
        #print(out);
        #tf.Print(out,[out]);
        #print(np.shape(out));
        return np.shape(out);