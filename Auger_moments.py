#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:17:27 2022
__________________________________________________________________

  "Learning the Composition of Ultra High Energy Cosmic Rays".
              https://arxiv.org/abs/2212.04760
__________________________________________________________________

Authors of the paper:
                      Blaz Bortolato,  
                      Jernej F. Kamenik,  
                      Michele Tammaro


Code used in the paper https://arxiv.org/abs/2212.04760:
"Learning the Composition of Ultra High Energy Cosmic Rays".



This script precomputes moments using experimental data, which must be provided
in the form of a table with 3 columns: 
    
    Energy [EeV], X_max[g/cm^2], d_X_max[g/cm^2].


"""



import os
import numpy as np
import matplotlib.pyplot as plt


output_dir = 'Preprocessed_data/'
sim_data_dir = 'Simulated_data_dir/'
aug_data_dir = 'Auger_data_dir/'


try:
    os.mkdir(output_dir)
except:
    None
    

#__________________________________________

data_aug = np.load('Auger_data_dir/hybrid_long_par_auger.npy')
E = data_aug[:,0]
dE = data_aug[:,1]
X = data_aug[:,2]
dX = data_aug[:,3]



def E_in_log(E):
    return np.log(E)/np.log(10) + 18

logE = E_in_log(E)
Energy_intervals = [[E_in_log(0.65), E_in_log(1.0), '0.65 EeV - 1 EeV'],
                    [E_in_log(1.0),  E_in_log(2.0),    '1 EeV - 2 EeV'],
                    [E_in_log(2.0),  E_in_log(5.0),    '2 EeV - 5 EeV'],
                    [17.8, 17.9, '17.8_17.9'],
                    [17.9, 18.0, '17.9_18.0'],
                    [18.0, 18.1, '18.0_18.1'],                    
                    ]


# OK

def non_central_moments(mu, sigma, n):
    '''
    Integral Normal(X | mu, sigma) X**n dX
    https://en.wikipedia.org/wiki/Normal_distribution
    OK
    '''
    if n == 0:
        return np.ones(len(mu))
    elif n == 1:
        return mu
    elif n == 2:
        return mu**2 + sigma**2
    elif n == 3:
        return mu**3 + 3*mu * sigma**2 
    elif n == 4:
        return mu**4 + 6*mu**2 * sigma**2 + 3 * sigma**4  





num_samples = 10**5
def compute_moments(logEmin, logEmax, E_interval):
    '''
        OK, OK
    '''
    
    con = (logEmin <= logE) & (logE < logEmax)

    X_temp  =  X[con]
    dX_temp = dX[con]
    N = len(X_temp)
    
    try:
        # asdf
        auger_moments = np.load(output_dir + '4moments_' + E_interval.replace(' ', '_') + '_AUGER.npy')
    
    except Exception as e:
        print(e)
        print('Start computing AUGER moments')
        RND = np.random.choice(N, replace = True, size = (num_samples, N))
        auger_moments = np.zeros((num_samples, 4))
        for n in range(1,5):
            pdfs_n = non_central_moments(mu = X_temp, sigma = dX_temp, n = n)
            auger_moments[:,n-1] = np.array([pdfs_n[RND[k]].sum()/N for k in range(num_samples)])
        
        np.save(output_dir + '4moments_' + E_interval.replace(' ', '_') + '_AUGER.npy', auger_moments)
        




for logEmin, logEmax, E_interval in Energy_intervals:
    compute_moments(logEmin, logEmax, E_interval)
    









