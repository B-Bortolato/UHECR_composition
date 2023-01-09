#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:23:28 2022


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

This script requires preprocessed simulated and experimantal data to be 
located in the folder "data_dir" - specified below. Preprocessed data 
are first computed using scripts:
    
    "Simulated_moments.py",
    
and

    "Auger_moments.py".


The script "Simulated_moments.py" requires simulated data to be given in a 
table with three columns: Energy [EeV], X_max [g/cm^2], d_Xmax [g/cm^2],
where d_Xmax is the systematic uncertainty on X_max obtained by fitting 
distribution of X.


Similarly, the script "Auger_moments.py" requires experimental data in
a table with columns:  Energy [EeV], X_max [g/cm^2], d_Xmax [g/cm^2].
In this case d_Xmax is the total systematic uncertainty on X_max.


Introduction to UltraNest can be found here: 
https://johannesbuchner.github.io/UltraNest/index.html.


The code is self-explanatory. In case of bugs please contact
blaz.bortolato@ijs.si.



"""

import os
import time 
import numpy as np
import matplotlib.pyplot as plt


import ultranest
import ultranest.stepsampler



#__________________________________configs
### configs



use_argparse = True

if use_argparse:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hadronic_model_index", help = "Possible values: 2, 4, 5, 6.", type = int)
    parser.add_argument("energy_bin_index", help = "Possible values: 0, 1, 2.", type = int)
    parser.add_argument("num_moments", help = "Number of moments, possible values: 1, 2, 3, 4.", type = int)
    parser.add_argument("live_points", help = "Number of live points, 400+.", type = int)


    args = parser.parse_args()
    hadronic_model_index = args.hadronic_model_index
    energy_bin_index = args.energy_bin_index
    num_moments = args.num_moments
    live_points = args.live_points

    
else:
    hadronic_model_index = 2
    energy_bin_index = 1
    num_moments = 3
    live_points = 400




"""
Dictonaries: "models" and "energy_intervals" are used to
load simulated and experimental data and determine the names
of files produced by this script (look at "signature").
"""


energy_interval = ['0.65 EeV - 1 EeV'.replace(' ', '_'),
                   '1 EeV - 2 EeV'.replace(' ', '_'),
                   '2 EeV - 5 EeV'.replace(' ', '_')][energy_bin_index]
                  
            


models = {2: 'EPOS',
          4: 'QGSJetII-04',
          5: 'QGSJet01',
          6: 'Sibyll',
          }   



#______________________________________________________________________
### version 
version = 'TEST'


model = models[had_model]
signature = model + '_' + energy_interval + '_num_moments_' + str(num_moments) + '_live_pts' + str(live_points) + '_version_' + version




"""Simulated and experimental data are located in "data_dir". Files produced by this script
are located in folder "Results", which is automatically created, if it does not exist."""


exp_name = signature
results_dir = 'Results/' + exp_name + '/'
data_dir = 'Preprocessed_data/'

try:
    os.mkdir('Results/')
except:
    None
    
try:
    os.mkdir(results_dir)
except:
    None


print('________________Parameters__________________')
print('Energy interval: ' + energy_interval)
print('Hadronic model: ' + model)
print('Number of moments: ' + str(num_moments))
print('Number of live points: ' + str(live_points))
print('dlogz = ' + str(dlogz))
print('Signature: ' + signature)
print('Directory: ' + results_dir)
print('____________________________________________')



#_______________________________________predifined functions
### predifined functions


D = 26
dlogz = 0.5 + 0.25*D


def dir_prior(quantiles):
    gamma_quantiles = -np.log(quantiles)
    return gamma_quantiles/gamma_quantiles.sum()



#_________________________________________________SIMULATED data
print('Load Simulated data')
G = np.load(data_dir + 'G_norm' + model + '_' +  energy_interval + '.npy')
Gn = G[:,:num_moments + 1]
# 26 x (D+1) x 10**5
# OK




def get_z_sim(w):
    ''' OK '''
    
    if num_moments == 1:
        norm = np.sum( Gn[:,0] * w.reshape(-1,1), axis = 0)
        Z_samples = np.sum( Gn[:,1] * w.reshape(-1,1), axis = 0)/norm   
        
        
    elif num_moments == 2:
        norm = np.sum( Gn[:,0] * w.reshape(-1,1), axis = 0)
        m1   = np.sum( Gn[:,1] * w.reshape(-1,1), axis = 0)/norm
        m2   = np.sum( Gn[:,2] * w.reshape(-1,1), axis = 0)/norm
        
        z1 = m1
        z2 = m2 - m1**2
        Z_samples = np.vstack([z1, z2]).T

    elif num_moments == 3:
        norm = np.sum( Gn[:,0] * w.reshape(-1,1), axis = 0)
        m1   = np.sum( Gn[:,1] * w.reshape(-1,1), axis = 0)/norm
        m2   = np.sum( Gn[:,2] * w.reshape(-1,1), axis = 0)/norm
        m3   = np.sum( Gn[:,3] * w.reshape(-1,1), axis = 0)/norm
        
        z1 = m1
        z2 = m2 - m1**2
        z3 = m3 - 3*m2*m1 + 2*m1**3
        Z_samples = np.vstack([z1, z2, z3]).T

    elif num_moments == 4:

        norm = np.sum( Gn[:,0] * w.reshape(-1,1), axis = 0)
        m1   = np.sum( Gn[:,1] * w.reshape(-1,1), axis = 0)/norm
        m2   = np.sum( Gn[:,2] * w.reshape(-1,1), axis = 0)/norm
        m3   = np.sum( Gn[:,3] * w.reshape(-1,1), axis = 0)/norm
        m4   = np.sum( Gn[:,4] * w.reshape(-1,1), axis = 0)/norm
        
        z1 = m1
        z2 = m2 - m1**2
        z3 = m3 - 3*m2*m1 + 2*m1**3
        z4 = m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4
        Z_samples = np.vstack([z1, z2, z3, z4]).T

    z_mean = np.mean(Z_samples, axis = 0)
    z_cov  = np.cov(Z_samples.T)
    return z_mean, z_cov
    


def get_z_sim_1moment(w):
    ''' OK '''
    norm = np.sum( Gn[:,0] * w.reshape(-1,1), axis = 0)
    Z_samples = np.sum( Gn[:,1] * w.reshape(-1,1), axis = 0)/norm   
        
    z_mean = np.mean(Z_samples)
    z_std  = np.std(Z_samples)
    return z_mean, z_std





def get_z_samples_sim(w):
    ''' OK '''
    
    
    norm = np.sum( G[:,0] * w.reshape(-1,1), axis = 0)
    m1   = np.sum( G[:,1] * w.reshape(-1,1), axis = 0)/norm
    m2   = np.sum( G[:,2] * w.reshape(-1,1), axis = 0)/norm
    m3   = np.sum( G[:,3] * w.reshape(-1,1), axis = 0)/norm
    m4   = np.sum( G[:,4] * w.reshape(-1,1), axis = 0)/norm
    
    z1 = m1
    z2 = m2 - m1**2
    z3 = m3 - 3*m2*m1 + 2*m1**3
    z4 = m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4
    Z_samples = np.vstack([z1, z2, z3, z4]).T
    return Z_samples




def get_z_samples_aug(nc_moments_aug):
    ''' OK '''
    m1, m2, m3, m4 = nc_moments_aug.T

    z1 = m1
    z2 = m2 - m1**2
    z3 = m3 - 3*m2*m1 + 2*m1**3
    z4 = m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4
    Z_samples = np.vstack([z1, z2, z3, z4]).T
    return Z_samples



#_________________________________________________AUGER data
print('Load Auger data')
nc_moments_aug = np.load(data_dir + '4moments_' + energy_interval + '_AUGER.npy')
Z_samples_aug = get_z_samples_aug(nc_moments_aug)[:,:num_moments]
# shape = (10**5, D)


if num_moments > 1:
    z_aug_mean = np.mean(Z_samples_aug, axis = 0)
    z_aug_cov  = np.cov(Z_samples_aug.T)
    inv_cov_aug = np.linalg.inv(z_aug_cov)

    
elif num_moments == 1:
    z_aug_mean = np.mean(Z_samples_aug)
    z_aug_std  = np.std(Z_samples_aug)
    z_aug_var = z_aug_std**2

# OK






def log_likelihood_1moment(w):
    '''Likelihood in case num_moments = 1.'''

    z_sim_mean, z_sim_std = get_z_sim_1moment(w)
    exp_1st_part = -1/2*(  (z_sim_mean/z_sim_std)**2 + (z_aug_mean/z_aug_std)**2  )
    exp_2nd_part =  1/2 * (z_sim_std * z_aug_std)**2 / (z_sim_std**2 + z_aug_std**2) * (z_sim_mean/z_sim_std**2 + z_aug_mean/z_aug_std**2)**2
    return -1/2 * np.log(2*np.pi * (z_sim_std**2 + z_aug_std**2)) + exp_1st_part + exp_2nd_part



def log_likelihood_xmoments(w):
    '''Likelihood in case num_moments > 1.'''

    z_sim_mean, z_sim_cov = get_z_sim(w)
    inv_cov_sim = np.linalg.inv(z_sim_cov)
    inv_inv_cov = np.linalg.inv(inv_cov_sim + inv_cov_aug)
    
    
    z_aux = inv_cov_sim.dot(z_sim_mean) +  inv_cov_aug.dot(z_aug_mean)
    
    
    norm = - 1/2*num_moments * np.log(2 *  np.pi)  - 1/2 * np.log(np.linalg.det( z_sim_cov + z_aug_cov))
    exp_1st_part = -1/2 * np.sum(inv_cov_sim.dot(z_sim_mean) * z_sim_mean)
    exp_2nd_part = -1/2 * np.sum(inv_cov_aug.dot(z_aug_mean) * z_aug_mean)
    exp_3rd_part =  1/2 * np.sum(inv_inv_cov.dot(z_aux) * z_aux)
    return norm + exp_1st_part + exp_2nd_part + exp_3rd_part




#_____________________________________________________________________
print('Start: sampling from posterior')
start_time = time.time()



param_names = ['w' + str(j) for j in range(D)]


if num_moments == 1: 
    Log_likelihood = log_likelihood_1moment
elif num_moments > 1:
    Log_likelihood = log_likelihood_xmoments


try:
    # Use this code to load compositions, weights and values of log-likleihood aftzer executing this script.
    # D = 26 is the number of primaries.
    d = np.load(results_dir + signature  + '_samples_weights_logL.npy')
    compositions = d[:,:D]
    weights = d[:,D]
    logL = d[:,D+1]
except Exception as e:
    print(e)
    
    
    sampler = ultranest.ReactiveNestedSampler(param_names, Log_likelihood, transform = dir_prior,
        log_dir = signature, # folder where to store files
        resume = True, # whether to resume from there (otherwise start from scratch)
        # warmstart_max_tau = 0.5,
        )
    
    
    
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                           nsteps = 100,
                           generate_direction = ultranest.stepsampler.generate_mixture_random_direction,
                           )
    
    
    result = sampler.run(
                       min_num_live_points = live_points,
                       dlogz = dlogz, # desired accuracy on logz
                       min_ess = live_points, # number of effective samples
                       max_num_improvement_loops = 3, # how many times to go back and improve
                      )
    
    
    compositions = result['weighted_samples']['points']
    weights      = result['weighted_samples']['weights']
    logL         = result['weighted_samples']['logl']
    
    
    np.save(results_dir + signature  + '_samples_weights_logL.npy', np.hstack([compositions, 
                                                                               weights.reshape(-1,1), 
                                                                               logL.reshape(-1,1)]))



print('UltraNest: done')
print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
start_time = time.time()
print('Start computing confidence intervals')
# OK






indices_sorted  = np.argsort(logL)
conf_lvls_sorted = 1 - weights[indices_sorted].cumsum()
comps_sorted = compositions[indices_sorted]
logL_sorted = logL[indices_sorted]
# OK


############################____PLOTS____############################
''' 
Unnecessary plots.
'''


xlabels = np.array(['Z = ' + str(j) for j in range(1, D+1)])
try:
    for j in range(D):
        plt.figure(figsize = (6,4))
        plt.scatter(compositions[:,j], logL, c = 'black', alpha = 0.2)
        plt.xlabel('fraction of ' + xlabels[j])
        plt.ylabel('logL')
        plt.title(signature)
        plt.ylim(np.max(logL)-16, np.max(logL) + 0.5)
        plt.tight_layout()
        plt.savefig(results_dir + signature + '_logL_' + xlabels[j]  + '.png')
        plt.close()
except Exception as e:
    print(e)


try:
    for j in range(D):
        plt.figure(figsize = (6,4))
        plt.scatter(comps_sorted[:,j], conf_lvls_sorted, c = 'black', alpha = 0.2)
        plt.xlabel('fraction of ' + xlabels[j])
        plt.ylabel('Confidence level')
        plt.title(signature)
        plt.tight_layout()
        plt.savefig(results_dir + signature + '_conf_lvl_' + xlabels[j]  + '.png')
        plt.close()
except Exception as e:
    print(e)




try:
    plt.figure(figsize = (6,4))
    plt.scatter(conf_lvls_sorted, logL_sorted, c = 'black', alpha = 0.5)    
    plt.xlabel('confidence level')
    plt.ylabel('logL')
    plt.title(signature)
    minlogL = logL_sorted[np.argmin( np.abs(conf_lvls_sorted - 0.975))]
    plt.ylim(minlogL, np.max(logL_sorted) + 0.5)
    plt.tight_layout()
    plt.savefig(results_dir + signature + '_logL_conf_lvls.png')
    plt.close()
except Exception as e:
    print(e)







def plot_4corr(Z_samples, color, alpha, axs):
    for i in range(4):
        axs[i, 0].set_ylabel('z' + str(i+1))
        for j in range(4):
            axs[i, j].scatter(Z_samples[:, j],
                              Z_samples[:, i], s = 1, c = color, alpha = alpha)
            if i == 3:
                axs[3, j].set_xlabel('z' + str(j+1))
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    return None




#################___Correlation_plots_w_best_estimate___################
Z_samples_aug = get_z_samples_aug(nc_moments_aug)
Z_samples_sim = get_z_samples_sim(w_best_estimate)


fig, axs = plt.subplots(nrows = 4, ncols = 4, figsize = (8, 8))
plot_4corr(Z_samples_aug, color = 'black', alpha = 1.0, axs = axs)
plot_4corr(Z_samples_sim, color = 'blue',  alpha = 1.0, axs = axs)
fig.suptitle('blue w_best, black: AUGER')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(results_dir + signature + '_4x4_corr_plots.png')
plt.show()
# OK






#####################[  CONFIDENCE INTERVALS ]#####################


def get_w_best():
    '''Estimates the best composition, returns composition, log-likelihood.'''

   nn = 5
   ref_logL = logL_sorted[-nn:]
   ref_comp = comps_sorted[-nn:]
   
   best_comp_array = [ ref_comp[-1] ]
   best_logL_array = [ ref_logL[-1] ]
       
   
   for j in range(nn):
       counts = 0
       core = ref_comp[j]
       best_logL = ref_logL[j]
       
       while counts < 10:
           
            samples__ = np.random.multinomial(1000, core, size = 200) + 0.00001
            samples__ = samples__/samples__.sum(axis = 1).reshape(-1,1)
            logl_samples = np.array([Log_likelihood(x) for x in samples__])
            
            if logl_samples.max() > best_logL:
                best_logL = logl_samples.max()
                core = samples__[np.argmax(logl_samples)]
                counts = 0
            else:
                counts += 1
       best_comp_array += [core]
       best_logL_array += [best_logL]
   return np.array(best_comp_array), np.array(best_logL_array)
            






def get_lower_bound(target_conf_lvl, z__):
    ''' 
    Estimated the lower bound for primary index z__ \in {0,1,...,25} 
    and target_conf_lvl \in (0,1).
    '''
    
   target_logL = logL_sorted[np.argmin( np.abs(conf_lvls_sorted - target_conf_lvl))]
   
   
   ref_comp = comps_sorted[logL_sorted > target_logL]
   sorted_ref_ind = np.argsort(ref_comp[:,z__])

   
   lower_bounds = []
   #_______________________________________lower bound
   for j in sorted_ref_ind[:3]:
       failed = 0
       counts = 0
       core_lower = ref_comp[j]
       lower_val = core_lower[z__]
       while counts < 10:
           
           for dummy_idx in range(5):
               samples__ = np.random.multinomial(1000, core_lower, size = 200) + 0.00001
               samples__ = samples__/samples__.sum(axis = 1).reshape(-1,1)
               logl_samples = np.array([Log_likelihood(x) for x in samples__])
               
               con = logl_samples > target_logL
               if con.sum() > 0:
                   break
               else:
                   failed += 1
           if failed >= 4:
               break
           
           valid_comp = samples__[con]
           core_new_lower = valid_comp[ np.argmin(valid_comp[:,z__]) ]
           if core_new_lower[z__] < lower_val:
               counts = 0
               lower_val = core_new_lower[z__]
               core_lower = core_new_lower

           else:
               counts += 1
       lower_bounds += [lower_val]
   return np.array(lower_bounds), core_lower, Log_likelihood(core_lower), target_logL







def get_upper_bound(target_conf_lvl, z__):
    target_logL = logL_sorted[np.argmin( np.abs(conf_lvls_sorted - target_conf_lvl))]

    
    ref_comp = comps_sorted[logL_sorted > target_logL]
    sorted_ref_ind = np.argsort(ref_comp[:,z__])

    
    upper_bounds = []
    
    #______________________________________upper bound
    for j in sorted_ref_ind[-3:]:
        failed = 0
        counts = 0
        core_upper = ref_comp[j]
        upper_val = core_upper[z__]
        while counts < 10:
            
            for dummy_idx in range(5):
                samples__ = np.random.multinomial(1000, core_upper, size = 200) + 0.00001
                samples__ = samples__/samples__.sum(axis = 1).reshape(-1,1)
                logl_samples = np.array([Log_likelihood(x) for x in samples__])
                
                con = logl_samples > target_logL
                if con.sum() > 0:
                    break
                else:
                    failed += 1
            if failed >= 4:
                break
            
            valid_comp = samples__[con]
            core_new_upper = valid_comp[ np.argmax(valid_comp[:,z__]) ]
            if core_new_upper[z__] > upper_val:
                counts = 0
                upper_val = core_new_upper[z__]
                core_upper = core_new_upper
            else:
                counts += 1
        upper_bounds += [upper_val]

    return np.array(upper_bounds), core_upper, Log_likelihood(core_upper), target_logL
    




def get_lower_bound_cumulative(target_conf_lvl, z__):
   target_logL = logL_sorted[np.argmin( np.abs(conf_lvls_sorted - target_conf_lvl))]
   
   
   ref_comp = comps_sorted[logL_sorted > target_logL]
   sorted_ref_ind = np.argsort(1 - ref_comp.cumsum(axis = 1)[:,z__])

   
   lower_bounds = []
   #_______________________________________lower bound
   for j in sorted_ref_ind[:3]:
       failed = 0
       counts = 0
       core_lower = ref_comp[j]
       lower_val = 1 - core_lower.cumsum()[z__]
       while counts < 10:
           
           for dummy_idx in range(5):
               samples__ = np.random.multinomial(1000, core_lower, size = 200) + 0.00001
               samples__ = samples__/samples__.sum(axis = 1).reshape(-1,1)
               logl_samples = np.array([Log_likelihood(x) for x in samples__])
               
               con = logl_samples > target_logL
               if con.sum() > 0:
                   break
               else:
                   failed += 1
           if failed >= 4:
               break
           
           valid_comp = samples__[con]
           valid_comp_cumulative = 1 - samples__[con].cumsum(axis = 1)
           idx = np.argmin(valid_comp_cumulative[:,z__])

           if valid_comp_cumulative[idx, z__] < lower_val:
               counts = 0
               lower_val = valid_comp_cumulative[idx, z__]
               core_lower = valid_comp[idx]
           else:
               counts += 1
               
       lower_bounds += [lower_val]
   return np.array(lower_bounds), core_lower, Log_likelihood(core_lower), target_logL







def get_upper_bound_cumulative(target_conf_lvl, z__):
    target_logL = logL_sorted[np.argmin( np.abs(conf_lvls_sorted - target_conf_lvl))]
    
    
    ref_comp = comps_sorted[logL_sorted > target_logL]
    sorted_ref_ind = np.argsort(1 - ref_comp.cumsum(axis = 1)[:,z__])

    
    upper_bounds = []
    
    #______________________________________upper bound
    for j in sorted_ref_ind[-3:]:
        failed = 0
        counts = 0
        core_upper = ref_comp[j]
        upper_val = 1 - core_upper.cumsum()[z__]
        while counts < 10:
            
            for dummy_idx in range(5):
                samples__ = np.random.multinomial(1000, core_upper, size = 200) + 0.00001
                samples__ = samples__/samples__.sum(axis = 1).reshape(-1,1)
                logl_samples = np.array([Log_likelihood(x) for x in samples__])
                
                con = logl_samples > target_logL
                if con.sum() > 0:
                    break
                else:
                    failed += 1
            if failed >= 4:
                break
            
            valid_comp = samples__[con]
            valid_comp_cumulative = 1 - valid_comp.cumsum(axis = 1)
            idx = np.argmax(valid_comp_cumulative[:,z__])            
            if valid_comp_cumulative[idx, z__] > upper_val:
                counts = 0
                upper_val = valid_comp_cumulative[idx, z__] 
                core_upper =  valid_comp[idx] 
            else:
                counts += 1
        upper_bounds += [upper_val]

    return np.array(upper_bounds), core_upper, Log_likelihood(core_upper), target_logL
    






#_____________________________________________confidence intervals I
''' Computes confidence intervals for all primaries for 1 sigma and 2 sigma confidence levels.'''

for target_conf_lvl in [0.683, 0.954]:

    print('computing bounds for conf. level = ' + str(target_conf_lvl))
        
    try:
        lower_bounds, upper_bounds = np.loadtxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_data.txt')
        for z__ in range(D):
            if lower_bounds[z__] < 0:
                print('z__ = ' + str(z__))
                print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
                start_time = time.time()


                conf_par = get_lower_bound(target_conf_lvl, z__)
                lower_bounds[z__] = conf_par[0].min()
                np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

                
                conf_par = get_upper_bound(target_conf_lvl, z__)
                upper_bounds[z__] = conf_par[0].max()
                np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

                
                np.savetxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_data.txt', np.vstack([lower_bounds, upper_bounds]))
                
    
    except Exception as e:
        print(e)
        lower_bounds = - np.ones(D)
        upper_bounds = - np.ones(D)
        

        for z__ in range(D):
            print('z__ = ' + str(z__))
            print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
            start_time = time.time()


            conf_par = get_lower_bound(target_conf_lvl, z__)
            lower_bounds[z__] = conf_par[0].min()
            np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

            
            conf_par = get_upper_bound(target_conf_lvl, z__)
            upper_bounds[z__] = conf_par[0].max()
            np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

            
            np.savetxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_data.txt', np.vstack([lower_bounds, upper_bounds]))

    

try:
    most_prob_data = np.loadtxt(results_dir + signature  + '_w_logL_most_probable.txt')
    w_most_probable = most_prob_data[:D]
    logL_most_probable = most_prob_data[D]
except Exception as e:
    print(e)
    print('Start computing w_best')
    print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
    start_time = time.time()
    
    best_comp_array, best_logl_array = get_w_best()
    idx = np.argmax(best_logl_array)
    w_most_probable = best_comp_array[idx]
    logL_most_probable = best_logl_array.max()
    np.savetxt(results_dir + signature  + '_w_logL_most_probable.txt', np.hstack([w_most_probable, logL_most_probable]))


print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
start_time = time.time()
# OK




###################################################################
# Unnecessary plot

plt.figure(figsize = (5,4))
colors = ['purple', 'navy']
target_conf_lvls = [0.954, 0.683]
labels = ['2 sigma', '1 sigma']
for k in range(len(target_conf_lvls)):
    target_conf_lvl = target_conf_lvls[k]
    lower_bounds, upper_bounds = np.loadtxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_data.txt')
    plt.fill_between(np.arange(D) + 1, lower_bounds, upper_bounds, step = 'mid', color = colors[k], label = labels[k])

plt.plot(np.arange(D)+1, w_most_probable, color = 'black', label = 'most probable', drawstyle = 'steps-mid')
plt.xlabel('Z')
plt.ylabel('fraction')
plt.title(signature + ' confidence intervals')
plt.legend()
plt.tight_layout()
plt.savefig(results_dir + signature + '_confidence_intervals.pdf')
plt.close()


print('Confidence intervals for composition: computed!')
print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
start_time = time.time()
print('Start computing confidence intervals for cumulative composition')
###################################################################




    


#_____________________________________________confidence intervals II [CUMULATIVE]
# for target_conf_lvl in [0.383, 0.683, 0.954]:
for target_conf_lvl in [0.683, 0.954]:
    print('computing bounds for conf. level = ' + str(target_conf_lvl))
        
    try:
        lower_bounds, upper_bounds = np.loadtxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_data.txt')
        for z__ in range(D):
            if lower_bounds[z__] < 0:
                print('z__ = ' + str(z__))
                print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
                start_time = time.time()


                conf_par = get_lower_bound_cumulative(target_conf_lvl, z__)
                lower_bounds[z__] = conf_par[0].min()
                np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

                
                conf_par = get_upper_bound_cumulative(target_conf_lvl, z__)
                upper_bounds[z__] = conf_par[0].max()
                np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

                
                np.savetxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_data.txt', np.vstack([lower_bounds, upper_bounds]))


                
    except Exception as e:
        print(e)
        lower_bounds = - np.ones(D)
        upper_bounds = - np.ones(D)
        

        for z__ in range(D):
            print('z__ = ' + str(z__))
            print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
            start_time = time.time()


            conf_par = get_lower_bound_cumulative(target_conf_lvl, z__)
            lower_bounds[z__] = conf_par[0].min()
            np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

            
            conf_par = get_upper_bound_cumulative(target_conf_lvl, z__)
            upper_bounds[z__] = conf_par[0].max()
            np.save(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_z__' + str(z__) + 'report.npy', np.array(conf_par, dtype = object))

            
            np.savetxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_data.txt', np.vstack([lower_bounds, upper_bounds]))





###################################################################
# Unnecessary plot

plt.figure(figsize = (5,4))
colors = ['purple', 'navy']
target_conf_lvls = [0.954, 0.683]
labels = ['2 sigma', '1 sigma']
for k in range(len(target_conf_lvls)):
    target_conf_lvl = target_conf_lvls[k]
    lower_bounds, upper_bounds = np.loadtxt(results_dir + signature  + '_' + str(target_conf_lvl) + 'conf_cumulative_data.txt')
    plt.fill_between(np.arange(D) + 1, lower_bounds, upper_bounds, step = 'mid', color = colors[k], label = labels[k])

plt.plot(np.arange(D) + 1, 1 - w_most_probable.cumsum(), color = 'black', label = 'most probable', drawstyle = 'steps-mid')
plt.xlabel('Z0')
plt.ylabel('fraction of primaries with Z > Z0')
plt.title(signature + ' confidence intervals')
plt.legend()
plt.tight_layout()
plt.savefig(results_dir + signature + '_cumulative_confidence_intervals.pdf')
plt.close()


print('Confidence intervals for cumulative composition: computed!')
print( 'time: ' + str(np.round( (time.time() - start_time)/3600, 2)) + ' hours')
print('DONE')









