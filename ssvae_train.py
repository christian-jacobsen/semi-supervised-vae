# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:41:10 2021

@author: Christian Jacobsen, University of Michigan
"""

from ssvae import *
from load_data_new import load_data_new
import torch
import numpy as np


def ssvae_load(path):
    config = torch.load(path)
    data_channels = 3
    initial_features = config['initial_features']
    growth_rate = config['growth_rate']
    n_latent = config['n_latent']
    dense_blocks = config['dense_blocks']
    VAE = ssvae(data_channels, initial_features, dense_blocks, growth_rate, n_latent)
    VAE.load_state_dict(config['model_state_dict'])
    loss_reg = config['l_reg']
    loss_rec = config['l_rec']
    beta_list = config['beta_final']
    #beta_list = 0
    return VAE, config

def ssvae_train(train_data_dir_u, train_data_dir_l, test_data_dir, save_dir, filename, \
                       epochs, rec_epochs, batch_size_u, batch_size_l, n_l, test_batch_size, \
                       wd, beta0, lr_schedule, nu, tau, \
                       data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                       prior, cont, cont_path):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
       
    train_loader_u, train_stats_u = load_data_new(train_data_dir_u, batch_size_u) # unlabeled data
    train_loader_l, train_stats_l = load_data_new(train_data_dir_l, batch_size_l, shuff = False) # labeled data    
    
    if cont:
        VAE, config = ssvae_load(cont_path)
    else:
        VAE = ssvae(data_channels, initial_features, dense_blocks, growth_rate, n_latent, prior)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr_schedule(0), weight_decay = wd)
    if cont:
        optimizer.load_state_dict(config['optimizer_state_dict'])    
    beta = beta0
    
    l_rec_list = np.zeros((epochs,))
    l_reg_list = np.zeros((epochs,))
    l_ss_list = np.zeros((epochs,))
    beta_list = np.zeros((epochs,))
    VAE.train()
    
    for epoch in range(epochs):
        if epoch % 10 == 0:
            print('=======================================')
            print('Epoch: ', epoch) 
        optimizer.param_groups[0]['lr'] = lr_schedule(epoch) #update learning rate
        
        for n, (_, _, y_u) in enumerate(train_loader_u):
            for m, (z_l, _, y_l) in enumerate(train_loader_l):
                y_u = y_u.to(device)
                y_l = y_l.to(device)
                z_l = z_l.to(device)
                y_l = y_l[0:n_l, :, :, :]
                z_l = z_l[0:n_l, :]
                if epoch==0: # compute initialized losses
                    _, _, _, _, _, l_rec, l_reg, l_ss = VAE.compute_loss(y_u, y_l, z_l)
                    l_rec_0 = torch.mean(l_rec)
                    l_reg_0 = torch.mean(l_reg)
                    l_ss_0 = torch.mean(l_ss) 
                
                VAE.zero_grad()
        
                _, _, _, _, _, l_rec, l_reg, l_ss = VAE.compute_loss(y_u, y_l, z_l)
                
                l_rec = torch.mean(l_rec)
                l_ss = torch.mean(l_ss)
                if epoch < rec_epochs:
                    if torch.mean(l_reg) > 1e10:
                        beta = 1
                    else:
                        beta = beta0
                    loss = l_rec + l_ss
                else:
                    beta = VAE.update_beta(beta, l_rec, nu, tau)
                    if beta > 1:
                        beta = 1
                    
                    loss = beta*(torch.mean(l_reg)) + l_ss + l_rec
                    
                loss.backward()
                optimizer.step()
                
                
                l_reg = torch.mean(l_reg)
                l_rec = l_rec.cpu().detach().numpy()
                l_reg = l_reg.cpu().detach().numpy()
                l_ss = l_ss.cpu().detach().numpy()
                
                l_rec_list[epoch] = l_rec
                l_reg_list[epoch] = l_reg
                l_ss_list[epoch] = l_ss
                beta_list[epoch] = beta
            if epoch % 10 == 0:
                print('beta = ', beta)
                print('l_rec = ', l_rec)
                print('l_reg = ', l_reg)   
                print('l_ss  = ', l_ss)
        
    for n, (true_params, _, true_data) in enumerate(train_loader_u):
        if n == 0:
            true_params = true_params.to(device)
            true_data = true_data.to(device)
            zmu, _, z, out_test, _ = VAE.forward(true_data)
            disentanglement_score = VAE.compute_dis_score(true_params, z)
            print(disentanglement_score)
    
    # we want to save the initialized losses also
    l_rec_list = np.insert(l_rec_list, 0, l_rec_0.cpu().detach().numpy())
    l_reg_list = np.insert(l_reg_list, 0, l_reg_0.cpu().detach().numpy())
    l_ss_list = np.insert(l_ss_list, 0, l_ss_0.cpu().detach().numpy())
    beta_list = np.insert(beta_list, 0, beta0)
     
    #save model
    config = {'train_data_dir_u': train_data_dir_u,
              'train_data_dir_l': train_data_dir_l,
              'test_data_dir': test_data_dir,
              'model': 'ssvae',
              'n_latent': n_latent,
              'dense_blocks': dense_blocks,
              'initial_features': initial_features,
              'growth_rate': growth_rate,
              'batch_size_u': batch_size_u,
              'batch_size_l': batch_size_l,
              'n_l': n_l,
              'test_batch_size': test_batch_size,
              'optimizer': optimizer,
              'prior': prior,
              'beta0': beta0,
              'nu': nu,
              'tau': tau,
              'rec_epochs': rec_epochs,
              'epochs': epochs,
              'dis_score': disentanglement_score,
              'l_reg': l_reg_list,
              'l_rec': l_rec_list,
              'l_ss': l_ss_list,
              'beta_final': beta_list,
              'model_state_dict': VAE.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(), 
              'weight_decay': wd
              }
    
    torch.save(config, save_dir + '/' + filename)
