#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:11:47 2022


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



This script inlcudes detector effects on simulated data and precomputes
moments used to infer the composition. For each primary Z = 1,...,26 
it requires a table with 3 columns: 
    
    Energy [EeV], X_max[g/cm^2], d_X_max[g/cm^2].


"""



import os
import numpy as np
import matplotlib.pyplot as plt


output_dir = 'Preprocessed_data/'
sim_data_dir = 'Simulated_data_dir/'


try:
    os.mkdir(output_dir)
    print(output_dir + '   is created')
except:
    None
    


#_____________________Choose hadronic model
had_model = 2



models = {2: 'EPOS',
          4: 'QGSJetII-04',
          5: 'QGSJet01',
          6: 'Sibyll',
          }   

#######################################################################
###################  MODIFY THE PART BELOW ############################
#######################################################################
''' 
Create array "data_full" with dimensions:
>>> data_full.shape
>>> (26, number of events, 3)

The third event for Oxygen (Z = 16) si given by:
>>> data_full[16-1][3-1]
>>> [Energy[EeV], Xmax[g/cm^2], dXmax[g/cm^2]]

'''


E_dict = {0 : '0.65 EeV - 1 EeV',
          1 : '1 EeV - 2 EeV',
          2 : '2 EeV - 5 EeV',
          }


#______________________Load SIMULATED data
data_E_idx0 = np.load(sim_data_dir + 'sim_xmax_data_' + str(had_model) + '_Eidx' + str(0) + '.npy', allow_pickle = True)
data_E_idx1 = np.load(sim_data_dir + 'sim_xmax_data_' + str(had_model) + '_Eidx' + str(1) + '.npy', allow_pickle = True)
data_E_idx2 = np.load(sim_data_dir + 'sim_xmax_data_' + str(had_model) + '_Eidx' + str(2) + '.npy', allow_pickle = True)



#_____________________all simulated data
data_full = []
for el in range(26):
    data_full += [np.vstack([ data_E_idx0[el], data_E_idx1[el], data_E_idx2[el] ])]

#######################################################################
###################  MODIFY THE PART ABOVE ############################
#######################################################################



def normal_pdf(x, mu, sigma):
    '''
    https://en.wikipedia.org/wiki/Normal_distribution
    OK
    '''
    return (2*np.pi)**(-1/2) * 1/sigma * np.exp(-0.5 * (x - mu)**2/sigma**2)




def epsilon_fun(Xmax, x1, x2, lam1, lam2):
    ''' 
    https://arxiv.org/pdf/1409.4809.pdf
    Eq. 7
    OK
    '''
    con1 = Xmax <= x1
    con2 = (x1 < Xmax) & (Xmax <= x2)
    con3 = Xmax > x2
    return np.exp(+(Xmax - x1)/lam1)*con1 + con2 + np.exp(-(Xmax - x2)/lam2)*con3 
    

def resolution_fun(Xrec, Xmax, sigma1, sigma2, f):
    '''
    https://arxiv.org/pdf/1409.4809.pdf
    Eq. 8
    OK
    '''
    return f * normal_pdf(Xrec - Xmax, 0, sigma1) + (1 - f)*normal_pdf(Xrec - Xmax, 0, sigma2)



# Direct copy from: https://arxiv.org/pdf/1409.4809.pdf  Table II
table_eff__ = [
#'lg E range   x1      λ1       x2      λ2'
'[17.8, 17.9) 586 ± 6 109 ± 17 881 ± 8 95 ± 7',
'[17.9, 18.0) 592 ± 9 133 ± 17 883 ± 8 101 ± 7',
'[18.0, 18.1) 597 ± 11 158 ± 19 885 ± 8 107 ± 7',
'[18.1, 18.2) 601 ± 14 182 ± 21 887 ± 8 113 ± 7',
'[18.2, 18.3) 604 ± 17 206 ± 24 888 ± 8 119 ± 7',
'[18.3, 18.4) 605 ± 20 230 ± 28 890 ± 8 125 ± 7',
'[18.4, 18.5) 605 ± 23 253 ± 32 892 ± 8 131 ± 7',
'[18.5, 18.6) 604 ± 27 276 ± 38 894 ± 9 137 ± 8',
'[18.6, 18.7) 602 ± 30 299 ± 44 896 ± 9 143 ± 8',
'[18.7, 18.8) 599 ± 33 321 ± 51 898 ± 9 150 ± 8',
'[18.8, 18.9) 594 ± 36 344 ± 59 899 ± 9 156 ± 8',
'[18.9, 19.0) 588 ± 39 365 ± 67 901 ± 9 162 ± 8',
'[19.0, 19.1) 581 ± 43 386 ± 77 903 ± 9 168 ± 8',
'[19.1, 19.2) 573 ± 46 407 ± 86 905 ± 9 174 ± 8',
'[19.2, 19.3) 563 ± 49 428 ± 98 907 ± 9 180 ± 8',
'[19.3, 19.4) 553 ± 52 447 ± 109 908 ± 9 186 ± 8',
'[19.4, 19.5) 540 ± 56 468 ± 122 910 ± 9 192 ± 8',
'[19.5, inf) 517 ± 62 502 ± 146 913 ± 10 203 ± 9',
]

# Reforming table above into a dictionary of arrays
table_eff = {}
for j in range(16):
    a = table_eff__[j].split(' ')
    logEmin__ = a[0].split('[')[1].split(',')[0]
    logEmax__ = a[1].split(')')[0]
    x1__      = float(a[2])
    lam1__    = float(a[5])
    x2__      = float(a[8])
    lam2__    = float(a[11])
    table_eff[logEmin__ + '_' + logEmax__] = [x1__, lam1__, x2__, lam2__]

# OK



# Direct copy from: https://arxiv.org/pdf/1409.4809.pdf  Table III
table_res__ = [
#lg E range     σ1        σ2          f
'[17.8, 17.9) 17.5 ± 0.7 33.7 ± 1.4 0.62',
'[17.9, 18.0) 16.7 ± 0.7 32.9 ± 1.4 0.63',
'[18.0, 18.1) 15.9 ± 0.7 31.9 ± 1.4 0.63',
'[18.1, 18.2) 15.1 ± 0.7 31.0 ± 1.4 0.64',
'[18.2, 18.3) 14.4 ± 0.7 30.0 ± 1.4 0.65',
'[18.3, 18.4) 13.8 ± 0.7 29.1 ± 1.5 0.66',
'[18.4, 18.5) 13.3 ± 0.7 28.1 ± 1.6 0.67',
'[18.5, 18.6) 12.8 ± 0.8 27.1 ± 1.6 0.68',
'[18.6, 18.7) 12.3 ± 0.8 26.3 ± 1.7 0.69',
'[18.7, 18.8) 12.0 ± 0.8 25.4 ± 1.8 0.70',
'[18.8, 18.9) 11.7 ± 0.9 24.7 ± 1.9 0.70',
'[18.9, 19.0) 11.5 ± 0.9 24.1 ± 1.9 0.71',
'[19.0, 19.1) 11.3 ± 0.9 23.6 ± 1.9 0.72',
'[19.1, 19.2) 11.2 ± 0.9 23.3 ± 2.0 0.73',
'[19.2, 19.3) 11.1 ± 0.9 23.1 ± 2.0 0.74',
'[19.3, 19.4) 11.1 ± 1.0 23.1 ± 2.0 0.75',
'[19.4, 19.5) 11.1 ± 1.0 23.2 ± 2.0 0.76',
'[19.5, inf) 11.2 ± 1.0 23.7 ± 2.1 0.77',
]


table_res = {}
for j in range(16):
    a = table_res__[j].split(' ')
    logEmin__ = a[0].split('[')[1].split(',')[0]
    logEmax__ = a[1].split(')')[0]
    sigma1__  = float(a[2])
    sigma2__  = float(a[5])
    f__       = float(a[8])
    table_res[logEmin__ + '_' + logEmax__] = [sigma1__, sigma2__, f__]


# OK




#######################################################################
###################  MODIFY THE PART BELOW ############################
#######################################################################
'''
'full_E_int' is the name of the energy interval, can be whatever.
'E_intervals' is an array of energy intervals in log-scale 
(given by https://arxiv.org/pdf/1409.4809.pdf) with base 10 that
are contained in 'full_E_int'.
'''

cases = [
        {'full_E_int' :   '17.8_17.9',
         'E_intervals' : ['17.8_17.9'],
         },
        
        {'full_E_int' :   '17.9_18.0',
         'E_intervals' : ['17.9_18.0'],
         },
        
        {'full_E_int' :  '18.0_18.1',
         'E_intervals' : ['18.0_18.1',],
         },
        
        {'full_E_int' :   '0.65 EeV - 1 EeV'.replace(' ', '_'),
         'E_intervals' : ['17.8_17.9', '17.9_18.0'],
         },
        
        {'full_E_int' :   '1 EeV - 2 EeV'.replace(' ', '_'),
         'E_intervals' : ['18.0_18.1', '18.1_18.2', '18.2_18.3'],
         },
        
        {'full_E_int' :   '2 EeV - 5 EeV'.replace(' ', '_'),
         'E_intervals' : ['18.3_18.4', '18.4_18.5', '18.5_18.6', '18.6_18.7'],
         },        
        ]

# OK
#######################################################################
###################  MODIFY THE PART ABOVE ############################
#######################################################################



def non_central_moments(X, sigma1, sigma2, f, n):
    '''
    Integral Resolution(Xr - X) Xr**n  dXr
    https://en.wikipedia.org/wiki/Normal_distribution
    OK
    '''
    if n == 0:
        return 1
    elif n == 1:
        return X
    elif n == 2:
        return X**2 +           f*sigma1**2 + (1 - f)*sigma2**2
    elif n == 3:
        return X**3 + 3*X*(     f*sigma1**2 + (1 - f)*sigma2**2  )
    elif n == 4:
        return X**4 + 6*X**2*(  f*sigma1**2 + (1 - f)*sigma2**2  ) + 3*(  f*sigma1**4 + (1 - f)*sigma2**4  )



# OK


####################[ New code ]####################
print(models[had_model])
Y = np.linspace(200, 1800, 4000 + 1)
dY = Y[1] - Y[0]
# OK


num_samples = 10**5
for case in cases:
    energy_range = case['full_E_int']
    energy_intervals = case['E_intervals']
    print(energy_range)
    
    try:
        Gnorm_Z_n_l = np.load(output_dir + 'G_norm' + models[had_model] + '_' +  energy_range + '.npy')
    except Exception as e:
        print(e)
        print('Start computing')
    
        
        Gnorm_Z_l_n = np.zeros((26, 5, num_samples))
        for z in range(26):
            print('Z = ' + str(z + 1))
            data_temp = data_full[z]
            logE = np.log(data_temp[:,0])/np.log(10) + 18
            
        
            
            pdf_X_n = []
            for energy_interval in energy_intervals:
                print(energy_interval)
                logEmin, logEmax = np.array(energy_interval.split('_'), dtype = float)
                x1, lam1, x2, lam2 = table_eff[energy_interval]
                sigma1, sigma2, f = table_res[energy_interval]
                
                
                con = (logEmin <= logE) & (logE < logEmax)
                xmax  = data_temp[con, 1]
                dxmax = data_temp[con, 2]
                
                
                #______________________I
                pdf_X_n_energy_bin = np.zeros((len(xmax), 5))
                for n in range(5):
                    detector_effects = non_central_moments(Y, sigma1 = sigma1, sigma2 = sigma2, f = f, n = n) * epsilon_fun(Y, x1 = x1, x2 = x2, lam1 = lam1, lam2 = lam2)
                    pdf_X_n_energy_bin[:,n] = np.array([ np.sum(normal_pdf(Y, mu = xmax[j], sigma = dxmax[j]) * detector_effects) * dY for j in range(len(xmax)) ])
                pdf_X_n += [pdf_X_n_energy_bin]
                
            pdf_X_n = np.vstack(pdf_X_n)
            N_Z = len(pdf_X_n)
            # OK
            
            #__________________________II
            RND = np.random.choice(N_Z, replace = True, size = (num_samples, N_Z))
            Gnorm_l_n = np.zeros((5, num_samples))
            for k in range(num_samples):
                Gnorm_l_n[:,k] = pdf_X_n[RND[k]].sum(axis = 0)
            Gnorm_Z_l_n[z] = Gnorm_l_n/N_Z
            # OK
            
        #______________________________III 
        np.save(output_dir + 'G_norm' + models[had_model] + '_' +  energy_range + '.npy', Gnorm_Z_l_n)
        # OK

