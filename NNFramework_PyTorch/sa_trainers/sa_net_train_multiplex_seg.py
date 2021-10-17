#import tensorflow as tf;
import torch.optim as optim
import os;
from distutils.util import strtobool;
import numpy as np;
import torch
import glob;
import skimage.io as io;

from .sa_net_train_multiplex import CNNTrainer;
#from sa_net_arch import AbstractCNNArch;
#from sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer_pytorch import OptimizerTypesPyTorch, CNNOptimizerPyTorch;
from ..sa_net_data_provider import AbstractDataProvider;
#import sa_helpers.multiplex_utils as multiplex_utils;


class MultiplexSegTrainer(CNNTrainer):
    def __init__(self, cnn_arch, cnn_arch_module, train_data_provider:AbstractDataProvider, validate_data_provider:AbstractDataProvider, optimizer_type, session_config, device, kwargs):
        # predefined list of arguments
        args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':256, 'epoch_size':10, 'display_step':5, 'save_best_only':False
            , 'with_features':'False', 'invert_out_img':'False', 'is_output_od':'False'};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        self.cnn_arch_module = cnn_arch_module;
        #self.cost_func = cost_func;
        self.train_data_provider = train_data_provider;
        self.validate_data_provider = validate_data_provider;
        self.optimizer_type = optimizer_type;
        #if(session_config == None):
        #    self.session_config = tf.ConfigProto();
        #else:
        #    self.session_config = session_config;
        self.device = device;
        self.max_epochs = int(args['max_epochs']);
        self.learning_rate = float(args['learning_rate']);
        self.batch_size = int(args['batch_size']);
        self.epoch_size = int(args['epoch_size']);
        self.display_step = int(args['display_step']);
        self.save_best_only = bool(strtobool(args['save_best_only']));
        self.with_features = bool(strtobool(args['with_features']));
        self.invert_out_img = bool(strtobool(args['invert_out_img']));
        self.is_output_od = bool(strtobool(args['is_output_od']));
        print('self.save_best_only = {}'.format(self.save_best_only) );
        #self.global_step = tf.Variable(0);

        #self.init = tf.global_variables_initializer();
        if(self.optimizer_type == OptimizerTypesPyTorch.ADAM):
            self.optimizer = CNNOptimizerPyTorch.adam_optimizer(self.learning_rate, self.cnn_arch, kwargs);
            #self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
            #self.optimizer.zero_grad();
        elif(self.optimizer_type == OptimizerTypesPyTorch.SGD):
            self.optimizer = CNNOptimizerPyTorch.sgd_optimizer(self.learning_rate, self.cnn_arch, kwargs);

        self.epoch_out_filename = os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_train_epoch_out.txt');
        self.minibatch_out_filename = os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_train_minibatch_out.txt');

    def train(self, do_init, do_restore, do_load_data):
        self.epoch_out_filewriter = open(self.epoch_out_filename, 'a+' );
        self.minibatch_out_filewriter = open(self.minibatch_out_filename, 'a+' );
        #with tf.Session(config=self.session_config) as sess:
        ##with tf.Session() as sess:
        #    with sess.as_default():
        #        if(do_init):
        #            sess.run(tf.global_variables_initializer());
        #            #sess.run(self.init);
        epoch_start_num = 1;
        if(do_restore):
            ##debug
            #print('attempt restore')
            checkpoint = self.cnn_arch_module.restore_model(is_test=False);            
            if(checkpoint is not None):
                #debug
                print('restore succeed')
                #print(self.optimizer);
                #print(checkpoint['optimizer_state_dict']);
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
                epoch_start_num = checkpoint['epoch'] + 1;
                #debug
                print('epoch_start_num ={}'.format(epoch_start_num ))
            ##debug
            #else:
            #    print('restore failed')

        if(do_load_data):
            self.train_data_provider.load_data();
            if(not (self.validate_data_provider is None)):
                self.validate_data_provider.load_data();

        if(self.epoch_size < 0)    :
            self.epoch_size = int(self.train_data_provider.data_count / self.batch_size + 0.5);
        if(not(self.validate_data_provider is None)):
            self.validate_epoch_size = int(self.validate_data_provider.data_count / self.batch_size + 0.5);
#        epoch_start_num = self.cnn_arch.epochs_count.eval();            

#        #tester_x, tester_label = self.train_data_provider.get_next_one();
        #tester_x = self.train_data_provider.get_next_one();
        #out_shape = self.cnn_arch.get_prediction_size(sess, tester_x);


        best_loss_saved_model_filename = None;
        last_saved_model_filename = None;

        best_validation_loss = float('inf')
        best_train_val_avg_accuracy = 0;

        current_validation_accuracy = None;    
        current_train_val_avg_accuracy = None;    

        for epoch in range(epoch_start_num, self.max_epochs+1):
            # run in training mode to ensure batch norm is calculated based on running mean and std
            self.cnn_arch_module.change_mode(is_test=False);
            total_loss = 0;
            total_count = 0;

            for step in range(0, self.epoch_size):
                batch_correct_count = 0;
                self.cnn_arch.zero_grad();
                self.optimizer.zero_grad();
                batch_x, batch_label = self.train_data_provider.get_next_n(self.batch_size);
                #print('batch_label.size() = ', batch_label.size())
                #print('batch_x.size() = ', batch_x.size())
                if(batch_x is None):
                    break;
                #print('train_x_min = ', batch_x.min());
                #print('train_x_max = ', batch_x.max());
                if(self.device is not None):
                    batch_x = batch_x.to(self.device);
                    batch_label = batch_label.to(self.device);
                #batch_label = batch_x; # for autoencoder only
                batch_y = self.cnn_arch.forward(batch_x);
                #if(isinstance(batch_y, tuple) and self.with_features):

                loss = self.cnn_arch_module.cost_func.calc_cost(batch_y, batch_label, self.device);

                ## debug
                #print('batch_y = {}, shape={}, type={}'.format(batch_y, batch_y.shape, batch_y.type()));
                loss.backward();
                self.optimizer.step()

                # clamp stain od values (weights)
                #self.cnn_arch_module.od_stains.clamp_(min=0, max=5.55);

                batch_count = batch_label.shape[0];
                total_count += batch_count ;

                if step % self.display_step == 0:
                    print('step = {}'.format(step));
                    self.output_minibatch_info(epoch, step, loss, batch_count)
                    #self.write_to_file(str(prediction), self.minibatch_out_filewriter);
                #break
                total_loss += loss.item();


            self.output_epoch_info(epoch, total_loss, self.epoch_size, total_count);                        
            
        
            # validate
            if (epoch % self.display_step) == 0:
                if(not(self.validate_data_provider is None)):
                    # run in test mode to ensure batch norm is calculated based on saved mean and std
                    self.cnn_arch_module.change_mode(is_test=True);
                    print("Running Validation:");
                    self.write_to_file("Running Validation"
                        , self.epoch_out_filewriter
                    );
                    self.validate_data_provider.reset();
                    validate_total_loss = 0;
                    validate_count = 0;
                    #for validate_step in range(0, self.validate_epoch_size):
                    for validate_step in range(0, self.validate_data_provider.data_count):
                        #validate_batch_x, validate_batch_label = self.validate_data_provider.get_next_n(self.batch_size);
                        validate_batch_x, validate_batch_label = self.validate_data_provider.get_next_n(1);
                        if(validate_batch_x is None):
                            break;
                        if(self.device is not None):
                            validate_batch_x = validate_batch_x.to(self.device) ;
                            validate_batch_label = validate_batch_label.to(self.device) ;
                        #validate_batch_label = validate_batch_x; # for autoencoder only
                        validate_batch_y = self.cnn_arch(validate_batch_x);
                        validate_loss = self.cnn_arch_module.cost_func.calc_cost(validate_batch_y, validate_batch_label, self.device);
                        validate_total_loss += validate_loss.item();
                        validate_count += validate_batch_y.shape[0] ;
                        if(validate_step <5):
                            validate_batch_y.detach().cpu().numpy().dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_pred_epoch_'+str(epoch)+'.npy'))
                            validate_batch_label.detach().cpu().numpy().dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_label_epoch_'+str(epoch)+'.npy'))
                            #self.output_sample_results_maxonly(epoch, validate_batch_x.detach().cpu().numpy(), validate_batch_y.detach().cpu().numpy(), validate_step)
                            ##self.output_stains(epoch, validate_batch_stains.detach().numpy());
                            ##self.output_stains(epoch, validate_batch_stains.detach().cpu().numpy()[0:3,:]);
                            #self.output_stains(epoch, validate_batch_stains.detach().cpu().numpy()[:,:]);
                            ###self.output_sample_results(epoch, validate_batch_label.detach().numpy(), validate_batch_y.detach().numpy());
                            ###self.output_sample_results(epoch, validate_batch_label.detach().cpu().numpy()
                            ###    , validate_batch_y.detach().cpu().numpy()
                            ###    , validate_batch_concentration.detach().cpu().numpy()
                            ###    , validate_batch_stains.detach().cpu().numpy()
                            ###);
                            ###self.output_sample_results_maxonly(epoch, validate_batch_label.detach().cpu().numpy()
                            ###    , validate_batch_y.detach().cpu().numpy()
                            ###    , validate_batch_concentration.detach().cpu().numpy()
                            ###    , validate_batch_stains.detach().cpu().numpy()[0:3,:]
                            ###);
                            #if(self.is_output_od):
                            #    self.output_sample_results_maxonly(epoch, validate_batch_label.detach().cpu().numpy()
                            #        , validate_batch_y.detach().cpu().numpy()
                            #        , validate_batch_concentration.detach().cpu().numpy()
                            #        , validate_batch_concentration_sm.detach().cpu().numpy()
                            #        #, validate_batch_stains.detach().cpu().numpy()[0:3,:]
                            #        , validate_batch_stains.detach().cpu().numpy()[:,:]
                            #    );
                            #else:
                            #    self.output_sample_results_rgb(epoch, validate_batch_label.detach().cpu().numpy()
                            #        , validate_batch_y.detach().cpu().numpy()
                            #        , validate_batch_concentration_sm.detach().cpu().numpy()
                            #        , validate_batch_stains.detach().cpu().numpy()[0:3,:]
                            #    );

                            #if(epoch == 0 or epoch % 5 != 0):
                            #    break; 
                    

                    self.output_val_epoch_info(epoch, validate_total_loss, validate_step+1, validate_count);                        
            
            
                saved = False;

                new_saved_model_filename = self.cnn_arch_module.save_model(None, self.optimizer, epoch);
                last_saved_model_filename = new_saved_model_filename;

            # permute the training data for the next epoch
            self.train_data_provider.reset(repermute=True);
            
        
#        print("Optimization Finished!")
        self.epoch_out_filewriter.close();
        self.minibatch_out_filewriter.close();


    def output_minibatch_info(self, epoch, batch, cost, total_count):
        print("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", total count= " + "{:d}".format(total_count) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.minibatch_out_filewriter
        );


    def output_epoch_info(self, epoch, total_cost, n_batches, total_count):
        print("\r\nepoch = " + str(epoch) \
            + ", training avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", training avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.epoch_out_filewriter
        );
        self.write_to_file("\r\n epoch = " + str(epoch) \
            + ", training avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.minibatch_out_filewriter
        );

    def output_val_epoch_info(self, epoch, total_cost, n_batches, total_count):
        print("\r\nepoch = " + str(epoch)  \
            + ", validation avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
        );
        self.write_to_file("epoch = " + str(epoch)   \
            + ", validation avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.epoch_out_filewriter
        );
        self.write_to_file("\r\n epoch = " + str(epoch)   \
            + ", validation avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.minibatch_out_filewriter
        );

    #def output_stains(self, epoch, stains_matrix_od):
    #    stains_matrix_od = stains_matrix_od.transpose();
    #    stains_matrix_od.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_od_epoch_'+str(epoch)+'.npy'));
    #    print("stains_matrix_od = ");
    #    print(stains_matrix_od);
    #    self.write_to_file("stains_matrix_od = " , self.epoch_out_filewriter);
    #    self.write_to_file(str(stains_matrix_od) , self.epoch_out_filewriter);
    #    stains_matrix_rgb = multiplex_utils.transform_optical_density_to_intensity(stains_matrix_od).astype(np.uint8);
    #    if(stains_matrix_rgb.shape[1] == 4):
    #        stains_matrix_rgb = multiplex_utils.transform_cmyk_to_rgb_1D(stains_matrix_rgb);
    #    print("stains_matrix_rgb = ");
    #    print(stains_matrix_rgb);
    #    stains_matrix_rgb.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_rgb_epoch_'+str(epoch)+'.npy'));
    #    self.write_to_file("stains_matrix_rgb = " , self.epoch_out_filewriter);
    #    self.write_to_file(str(stains_matrix_rgb) , self.epoch_out_filewriter);
    #    # save stains visualization image
    #    square_side = 32;
    #    stains_visualize = np.zeros((square_side, stains_matrix_rgb.shape[0]*square_side, 3), dtype=np.uint8);   
    #    for i in range(stains_matrix_rgb.shape[0]):
    #        stains_visualize[:, i*square_side:(i+1)*square_side] =  stains_matrix_rgb[i];
    #    if(self.invert_out_img):
    #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_vis_epoch_'+str(epoch)+'.png'), 255-stains_visualize);
    #    else:
    #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_vis_epoch_'+str(epoch)+'.png'), stains_visualize);


    ##def output_sample_results(self, epoch, batch_x, batch_y, batch_concentrations, stains):
    ##    batch_x_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_x).astype(np.uint8);
    ##    batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
    ##    batch_y_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_y).astype(np.uint8);
    ##    batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
    ##    print('batch_x_rgb.size() = ', batch_x_rgb.shape)
    ##    print('batch_y_rgb.size() = ', batch_y_rgb.shape)
    ##    print('batch_concentrations.size() = ', batch_concentrations.shape)
    ##    print('stains.size() = ', stains.shape)

    ##    for i in range(batch_x_rgb.shape[0]):
    ##        # save each image and corresponding reconstruction in batch
    ##        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'.png'), batch_x_rgb[i]);
    ##        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out.png'), batch_y_rgb[i]);

    ##        # save each stain estimate image
    ##        for s in range(stains.shape[1]):
    ##            # debug
    ##            #print('batch_concentrations[i][s].shape = ', batch_concentrations[i][s].shape);
    ##            stain_conc_img_od = np.matmul(stains[:,s].reshape((-1,1)), batch_concentrations[i][s].reshape(1,-1));
    ##            # debug
    ##            #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    ##            #print('stains[:,s].min() = ', stains[:,s].min());
    ##            #print('stains[:,s].max() = ', stains[:,s].max());
    ##            #print('batch_concentrations[i][s].min() = ', batch_concentrations[i][s].min());
    ##            #print('batch_concentrations[i][s].max() = ', batch_concentrations[i][s].max());
    ##            #print('stain_conc_img_od.min() = ', stain_conc_img_od.min());
    ##            #print('stain_conc_img_od.max() = ', stain_conc_img_od.max());
    ##            stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
    ##            # debug
    ##            #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    ##            stain_conc_img_od = stain_conc_img_od.reshape((batch_x_rgb.shape[1],batch_x_rgb.shape[2],3))
    ##            # debug
    ##            #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    ##            stain_conc_img_rgb = multiplex_utils.transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
    ##            # debug
    ##            #print('stain_conc_img_rgb.min() = ', stain_conc_img_rgb.min());
    ##            #print('stain_conc_img_rgb.max() = ', stain_conc_img_rgb.max());
    ##            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out.png'), stain_conc_img_rgb);

    #def output_sample_results_maxonly(self, epoch, batch_x, batch_y, batch_concentrations, batch_concentrations_sm, stains):
    #    #batch_x_rgb = multiplex_utils.transform_optical_density_to_intensity((batch_x/2+0.5)*4).astype(np.uint8);
    #    const_val = 255.0;
    #    batch_x_rgb = batch_x.astype(np.uint8);
    #    print('batch_x_rgb.shape = ', batch_x_rgb.shape);
    #    if(batch_x_rgb.shape[1]<20):
    #        batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
    #    if(batch_x_rgb.shape[-1] == 4):
    #        for i in range(batch_x_rgb.shape[0]):
    #            batch_x_rgb[i,:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(batch_x_rgb[i]);
    #            batch_x_rgb[i,:,:,3] = 255
    #        #const_val = 100.0;
    #    elif(batch_x_rgb.shape[-1]>4):
    #        batch_x_rgb = batch_x_rgb[:,:,:,:3];
    #    batch_y_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_y, const_val=const_val).astype(np.uint8);
    #    batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
    #    if(batch_y_rgb.shape[-1] == 4):
    #        for i in range(batch_x_rgb.shape[0]):
    #            batch_y_rgb[i,:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(batch_y_rgb[i]); 
    #            batch_y_rgb[i,:,:,3] = 255;
    #    print('batch_x_rgb.size() = ', batch_x_rgb.shape)
    #    print('batch_y_rgb.size() = ', batch_y_rgb.shape)
    #    print('batch_concentrations.size() = ', batch_concentrations.shape)
    #    print('stains.size() = ', stains.shape)
    #    batch_concentrations.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_conc_epoch_'+str(epoch)+'.npy'))
    #    s_sum = stains.sum(axis=0)
    #    s_sum_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_sum,axis=0),axis=2),axis=3)
    #    b=np.multiply(s_sum_expand,batch_concentrations)

    #    s_max = stains.max(axis=0)
    #    s_max_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_max,axis=0),axis=2),axis=3)
    #    b_max = np.multiply(s_max_expand,batch_concentrations)

    #    #batch_concentrations_flatten = batch_concentrations.reshape((-1, batch_concentrations.shape[1], batch_concentrations.shape[2]*batch_concentrations.shape[3]));
    #    #batch_concentrations_flatten_argmax = batch_concentrations_flatten.argmax(axis=1);
    #    #one_hot =  np.zeros(batch_concentrations_flatten.shape);
    #    #print('batch_concentrations_flatten.shape() = ', batch_concentrations_flatten.shape)
    #    #print('batch_concentrations_flatten_argmax.shape() = ', batch_concentrations_flatten_argmax.shape)
    #    #print('batch_concentrations_flatten_argmax.min() = ', batch_concentrations_flatten_argmax.min())
    #    #print('batch_concentrations_flatten_argmax.max() = ', batch_concentrations_flatten_argmax.max())
    #    #print('one_hot.shape() = ', one_hot.shape)
    #    #one_hot[np.arange(one_hot.shape[0]),batch_concentrations_flatten_argmax,np.arange(one_hot.shape[2])]=1;
    #    #one_hot_sum = one_hot.sum(axis=1);
    #    #print('one_hot_sum = ', one_hot_sum)
    #    #print('one_hot_sum.min() = ', one_hot_sum.min())
    #    #print('one_hot_sum.max() = ', one_hot_sum.max())
    #    #print('one_hot = ', one_hot)
    #    #one_hot[:,0,:] = 1
    #    #one_hot_sum = one_hot.sum(axis=1);
    #    #print('one_hot_sum = ', one_hot_sum)
    #    #print('one_hot_sum.min() = ', one_hot_sum.min())
    #    #print('one_hot_sum.max() = ', one_hot_sum.max())
    #    #new_concentration_flatten = one_hot * batch_concentrations_flatten;
    #    #print('new_concentration.shape() = ', new_concentration_flatten.shape)
    #    #new_concentration = new_concentration_flatten.reshape(batch_concentrations.shape)
    #    #new_out_od = np.matmul(stains, new_concentration_flatten);
    #    #print('new_out_od.shape() = ', new_out_od.shape)
    #    #new_out_rgb = self.transform_optical_density_to_intensity(new_out_od).astype(np.uint8);
    #    #new_out_rgb = new_out_rgb.reshape(batch_y.shape);
    #    #new_out_rgb = np.transpose(new_out_rgb, axes=(0,2,3,1))
    #    ##print('one_hot = ', one_hot)

    #    #for i in range(batch_x_rgb.shape[0]):
    #    for i in range(min(batch_x_rgb.shape[0], 5)):
    #        batch_concentrations_flatten = batch_concentrations[i].reshape((batch_concentrations.shape[1], batch_concentrations.shape[2]*batch_concentrations.shape[3]));
    #        batch_concentrations_flatten_argmax = batch_concentrations_flatten.argmax(axis=0);
    #        one_hot =  np.zeros(batch_concentrations_flatten.shape);

    #        b_flatten = b[i].reshape((b.shape[1], b.shape[2]*b.shape[3]));
    #        b_flatten_argmax = b_flatten.argmax(axis=0);
    #        b_one_hot =  np.zeros(b_flatten.shape);

    #        #b_max_flatten = b_max[i].reshape((b_max.shape[1], b_max.shape[2]*b_max.shape[3]));
    #        #b_max_flatten_argmax = b_max_flatten.argmax(axis=0);
    #        #b_max_one_hot =  np.zeros(b_max_flatten.shape);

    #        # debug
    #        #print('batch_concentrations_flatten.shape() = ', batch_concentrations_flatten.shape)
    #        #print('batch_concentrations_flatten_argmax.shape() = ', batch_concentrations_flatten_argmax.shape)
    #        #print('batch_concentrations_flatten_argmax.min() = ', batch_concentrations_flatten_argmax.min())
    #        #print('batch_concentrations_flatten_argmax.max() = ', batch_concentrations_flatten_argmax.max())
    #        #print('one_hot.shape() = ', one_hot.shape)
    #        one_hot[batch_concentrations_flatten_argmax,np.arange(one_hot.shape[1])]=1;
    #        one_hot_sum = one_hot.sum(axis=1);

    #        b_one_hot[b_flatten_argmax,np.arange(b_one_hot.shape[1])]=1;

    #        #b_max_one_hot[b_max_flatten_argmax,np.arange(b_max_one_hot.shape[1])]=1;

    #        # debug
    #        #print('one_hot_sum = ', one_hot_sum)
    #        #print('one_hot_sum.min() = ', one_hot_sum.min())
    #        #print('one_hot_sum.max() = ', one_hot_sum.max())
    #        #print('one_hot = ', one_hot)
    #        #one_hot[0,:] = 1
    #        #one_hot_sum = one_hot.sum(axis=1);

    #        # debug
    #        #print('one_hot_sum = ', one_hot_sum)
    #        #print('one_hot_sum.min() = ', one_hot_sum.min())
    #        #print('one_hot_sum.max() = ', one_hot_sum.max())
    #        new_concentration_flatten = one_hot * batch_concentrations_flatten;
    #        b_new_concentration_flatten = b_one_hot * batch_concentrations_flatten;
    #        #b_max_new_concentration_flatten = b_max_one_hot * batch_concentrations_flatten;
    #        # debug
    #        #print('new_concentration.shape() = ', new_concentration_flatten.shape)
    #        new_concentration = new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
    #        new_out_od = np.matmul(stains, new_concentration_flatten);
    #        new_out_od_mask = np.matmul(stains, one_hot);

    #        b_new_concentration = b_new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
    #        b_new_out_od = np.matmul(stains, b_new_concentration_flatten);
    #        b_new_out_od_mask = np.matmul(stains, b_one_hot);

    #        #b_max_new_concentration = b_max_new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
    #        #b_max_new_out_od = np.matmul(stains, b_max_new_concentration_flatten);

    #        # debug
    #        print('new_out_od.shape() = ', new_out_od.shape)
    #        print('stains.shape = ', stains.shape)
    #        new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(new_out_od, const_val=const_val).astype(np.uint8);
    #        new_out_rgb = new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
    #        new_out_rgb = np.transpose(new_out_rgb, axes=(1,2,0))
    #        if(new_out_rgb.shape[-1] == 4):
    #            print('transform_cmyk_to_rgb')
    #            new_out_rgb[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(new_out_rgb);
    #            new_out_rgb[:,:,3]  = 255;
    #        print('after  transform_cmyk_to_rgb xxxxxxxxxxxx')

    #        print('new_out_od_mask.shape() = ', new_out_od_mask.shape)
    #        new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(new_out_od_mask, const_val=const_val).astype(np.uint8);
    #        new_out_rgb_mask = new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
    #        new_out_rgb_mask = np.transpose(new_out_rgb_mask, axes=(1,2,0))
    #        if(new_out_rgb_mask.shape[-1] == 4):
    #            new_out_rgb_mask[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(new_out_rgb_mask);
    #            new_out_rgb_mask[:,:,3]  = 255;

    #        b_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od, const_val=const_val).astype(np.uint8);
    #        b_new_out_rgb = b_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
    #        b_new_out_rgb = np.transpose(b_new_out_rgb, axes=(1,2,0))
    #        if(b_new_out_rgb.shape[-1] == 4):
    #            b_new_out_rgb[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(b_new_out_rgb);
    #            b_new_out_rgb[:,:,3]  = 255;

    #        b_new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od_mask, const_val=const_val).astype(np.uint8);
    #        b_new_out_rgb_mask = b_new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
    #        b_new_out_rgb_mask = np.transpose(b_new_out_rgb_mask, axes=(1,2,0))
    #        if(b_new_out_rgb_mask.shape[-1] == 4):
    #            b_new_out_rgb_mask[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(b_new_out_rgb_mask);
    #            b_new_out_rgb_mask[:,:,3]  = 255;

    #        #b_max_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_max_new_out_od).astype(np.uint8);
    #        #b_max_new_out_rgb = b_max_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
    #        #b_max_new_out_rgb = np.transpose(b_max_new_out_rgb, axes=(1,2,0))

    #        # save each image and corresponding reconstruction in batch
    #        if(self.invert_out_img):
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'.png'), 255-batch_x_rgb[i]);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out.png'), 255-batch_y_rgb[i]);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly.png'), 255-new_out_rgb);
    #        else:
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'.png'), batch_x_rgb[i]);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out.png'), batch_y_rgb[i]);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly.png'), new_out_rgb);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly_b.png'), b_new_out_rgb);
    #            #io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly_b_max.png'), b_max_new_out_rgb);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly_mask.png'), new_out_rgb_mask);
    #            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out_maxonly_b_mask.png'), b_new_out_rgb_mask);


    #        ## save each stain estimate image
    #        #for s in range(stains.shape[1]):
    #        #    # debug
    #        #    #print('batch_concentrations[i][s].shape = ', batch_concentrations[i][s].shape);
    #        #    stain_conc_img_od = np.matmul(stains[:,s].reshape((-1,1)), batch_concentrations[i][s].reshape(1,-1));
    #        #    stain_conc_img_od_maxonly = np.matmul(stains[:,s].reshape((-1,1)), new_concentration[s].reshape(1,-1));
    #        #    b_stain_conc_img_od_maxonly = np.matmul(stains[:,s].reshape((-1,1)), b_new_concentration[s].reshape(1,-1));
    #        #    b_max_stain_conc_img_od_maxonly = np.matmul(stains[:,s].reshape((-1,1)), b_max_new_concentration[s].reshape(1,-1));
    #        #    # debug
    #        #    #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    #        #    #print('stains[:,s].min() = ', stains[:,s].min());
    #        #    #print('stains[:,s].max() = ', stains[:,s].max());
    #        #    #print('batch_concentrations[i][s].min() = ', batch_concentrations[i][s].min());
    #        #    #print('batch_concentrations[i][s].max() = ', batch_concentrations[i][s].max());
    #        #    #print('stain_conc_img_od.min() = ', stain_conc_img_od.min());
    #        #    #print('stain_conc_img_od.max() = ', stain_conc_img_od.max());
    #        #    stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
    #        #    stain_conc_img_od_maxonly = np.transpose(stain_conc_img_od_maxonly, axes=(1,0))
    #        #    b_stain_conc_img_od_maxonly = np.transpose(b_stain_conc_img_od_maxonly, axes=(1,0))
    #        #    b_max_stain_conc_img_od_maxonly = np.transpose(b_max_stain_conc_img_od_maxonly, axes=(1,0))
    #        #    # debug
    #        #    #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    #        #    stain_conc_img_od = stain_conc_img_od.reshape((batch_x_rgb.shape[1],batch_x_rgb.shape[2],3))
    #        #    stain_conc_img_od_maxonly = stain_conc_img_od_maxonly.reshape((batch_x_rgb.shape[1],batch_x_rgb.shape[2],3))
    #        #    b_stain_conc_img_od_maxonly = b_stain_conc_img_od_maxonly.reshape((batch_x_rgb.shape[1],batch_x_rgb.shape[2],3))
    #        #    b_max_stain_conc_img_od_maxonly = b_max_stain_conc_img_od_maxonly.reshape((batch_x_rgb.shape[1],batch_x_rgb.shape[2],3))
    #        #    # debug
    #        #    #print('stain_conc_img_od.shape = ', stain_conc_img_od.shape);
    #        #    stain_conc_img_rgb = multiplex_utils.transform_optical_density_to_intensity(stain_conc_img_od, const_val=const_val).astype(np.uint8);
    #        #    stain_conc_img_rgb_maxonly = multiplex_utils.transform_optical_density_to_intensity(stain_conc_img_od_maxonly, const_val=const_val).astype(np.uint8);
    #        #    b_stain_conc_img_rgb_maxonly = multiplex_utils.transform_optical_density_to_intensity(b_stain_conc_img_od_maxonly, const_val=const_val).astype(np.uint8);
    #        #    b_max_stain_conc_img_rgb_maxonly = multiplex_utils.transform_optical_density_to_intensity(b_max_stain_conc_img_od_maxonly, const_val=const_val).astype(np.uint8);
    #        #    # debug
    #        #    #print('stain_conc_img_rgb.min() = ', stain_conc_img_rgb.min());
    #        #    #print('stain_conc_img_rgb.max() = ', stain_conc_img_rgb.max());
    #        #    if(self.invert_out_img):
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out.png'), 255-stain_conc_img_rgb);
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out_maxonly.png'), 255-stain_conc_img_rgb_maxonly);
    #        #    else:
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out.png'), stain_conc_img_rgb);
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out_maxonly.png'), stain_conc_img_rgb_maxonly);
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out_maxonly_b.png'), b_stain_conc_img_rgb_maxonly);
    #        #        io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+ '_stain_'+ str(s)+'_out_maxonly_b_max.png'), b_max_stain_conc_img_rgb_maxonly);


    def output_sample_results_maxonly(self, epoch, batch_x, batch_y, sample_indx):
        print('batch_x',batch_x.shape)
        if(batch_x.shape[1]<20):
            batch_x = np.transpose(batch_x, axes=(0,2,3,1))
        print('batch_x',batch_x.shape)
        print('batch_y',batch_y.shape)
        #for i in range(batch_x_rgb.shape[0]):
        for i in range(min(batch_x.shape[0], 5)):
            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(sample_indx*batch_x.shape[0]+i)+'.png'), batch_x[i]);
            batch_y_argmax = batch_y[i].squeeze().argmax(axis=0)
            print('batch_y[i].squeeze()',batch_y[i].squeeze().shape)
            print('batch_y[i].squeeze().argmax(axis=0)',batch_y[i].squeeze().argmax(axis=0).shape)
            out_im = np.ones((batch_y_argmax.shape[0], batch_y_argmax.shape[1],3))
            
            out_im[batch_y_argmax==0] = (151, 82, 62)
            out_im[batch_y_argmax==1] = (118, 62, 151)
            out_im[batch_y_argmax==2] = (30, 30, 30)
            out_im[batch_y_argmax==3] = (62, 147, 151)
            out_im[batch_y_argmax==4] = (165, 168, 45)
            out_im[batch_y_argmax==5] = (174, 38, 75)
            out_im[batch_y_argmax==6] = (62, 104, 151)
            out_im[batch_y_argmax==7] = (212, 212, 210)

            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(sample_indx*batch_x.shape[0]+i)+'_seg.png'), out_im.astype(np.uint8));

    def write_to_file(self, text, filewriter):
        filewriter.write('\r\n');
        filewriter.write(text);
        filewriter.flush();

    def print_optimizer_params(self):
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])




    def delete_model_files(self, filepath):
        if(filepath is None):
            return;
        filepath, _ = os.path.splitext(filepath);
        print('delete_model_files = ', filepath)
        file_pattern = filepath + '*';
        files = glob.glob(file_pattern);
        for file in files: 
            print(file);
            os.remove(file);

    #def transform_optical_density_to_intensity(self, od):    
    #    print('transform_optical_density_to_intensity')
    #    rgb = np.exp(-od)*255.0 ###
    #    return rgb ;
