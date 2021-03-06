"""
@author: Christian Jacobsen, University of Michigan

VAE training file: requires a configuration file "config_train.py" to train

"""

from config_train import *
from ssvae_train import *
import os
import time
import numpy as np



if __name__ == '__main__':
    
    start = time.time()
    n_l_v = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] 
    save_dir0 = save_dir    
    for n_l in n_l_v:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
            
        
        file_exists = True
        trial_num = 0
        while file_exists:
            if trial_num > 100:
                break
            save_dir_temp = save_dir + '/VAE_' + str(trial_num)
            filename = 'VAE_' + str(trial_num) + '.pth'
            path_exists = os.path.exists(save_dir_temp)
            if path_exists:
                file_exists = os.path.exists(save_dir_temp + '/' + filename)
            else:
                file_exists = False
                os.mkdir(save_dir_temp)
            trial_num += 1
        
        save_dir = save_dir_temp
        
        ssvae_train(train_data_dir_u, train_data_dir_l, test_data_dir, save_dir, filename, \
                                 epochs, rec_epochs, batch_size_u, batch_size_l, n_l, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                                 data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                                 prior, continue_training, continue_path)
        save_dir = save_dir0 
    end = time.time()
    
    print('------------ Training Completed --------------')
    print('Elapsed Time: ', (end-start)/60, ' mins')
    print('Save Location: ', save_dir)



