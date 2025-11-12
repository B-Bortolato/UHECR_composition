#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 23:01:55 2025

@author: blaz



****************************************************************
         Includes detector effects on simulated data
****************************************************************

Inputs (simulated data): 
    - E (EeV), 
    - Xmax (g/cm^2), 
    - dXmax (g/cm^2)
    
Outputs (include detector effects): 
    - weight of the event (/), 
    - Xmax (g/cm^2), 
    - dXmax (g/cm^2), 
    - log_10(E / eV )


Based on paper: https://arxiv.org/pdf/1409.4809.pdf


"""



import numpy as np



#________________________________Detector corrections
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


def generate_table_xlambda():
    # Direct copy (to avoid errors) from: https://arxiv.org/pdf/1409.4809.pdf  Table II
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
    '[19.5, 1000) 517 ± 62 502 ± 146 913 ± 10 203 ± 9',
    ]
    
    # Reforming table above into a dictionary of arrays
    table_eff = {}
    for j in range(18):
        a = table_eff__[j].split(' ')
        logEmin__ = a[0].split('[')[1].split(',')[0]
        logEmax__ = a[1].split(')')[0]
        x1__      = float(a[2])
        lam1__    = float(a[5])
        x2__      = float(a[8])
        lam2__    = float(a[11])
        table_eff[logEmin__ + '_' + logEmax__] = [x1__, lam1__, x2__, lam2__]
    return table_eff




def generate_table_sigmaf():
    # Direct copy (to avoid errors) from: https://arxiv.org/pdf/1409.4809.pdf  Table III
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
    '[19.5, 1000) 11.2 ± 1.0 23.7 ± 2.1 0.77',
    ]
    
    
    table_res = {}
    for j in range(18):
        a = table_res__[j].split(' ')
        logEmin__ = a[0].split('[')[1].split(',')[0]
        logEmax__ = a[1].split(')')[0]
        sigma1__  = float(a[2])
        sigma2__  = float(a[5])
        f__       = float(a[8])
        table_res[logEmin__ + '_' + logEmax__] = [sigma1__, sigma2__, f__]
    return table_res




def non_central_moments(mu, sigma1, sigma2, f, n):
    '''
    Integral Resolution(Xr - X) Xr**n  dXr
    https://en.wikipedia.org/wiki/Normal_distribution
    Wolfram alpha check
    '''
    s2 = f*sigma1**2 + (1 - f)*sigma2**2
    s4 = f*sigma1**4 + (1 - f)*sigma2**4

    if n == 0:
        return 1
    
    elif n == 1:
        return mu
    
    elif n == 2:
        return mu**2 + s2
    
    elif n == 3:
        return mu**3 + 3*mu*s2
    
    elif n == 4:
        return mu**4 + 6*mu**2*s2 + 3*s4



def central_moments(mu, sigma1, sigma2, f, n):
    '''
    Integral Resolution(Xr - X) Xr**n  dXr
    https://en.wikipedia.org/wiki/Normal_distribution
    Wolfram alpha check
    '''

    if n == 0:
        return 1
    
    elif n == 1:
        return mu
    
    elif n == 2:
        return f*sigma1**2 + (1 - f)*sigma2**2








class DetectorEffects:
    """
    A class for applying detector efficiency and resolution effects
    to simulated data UHECR.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list containing one array per primary type:
        data = [primary1, primary2, primary3, ...]

        Each primary array must have shape (n_events, 3), with columns:

            E [EeV] | Xmax [g/cm^2] | dXmax [g/cm^2]

        Example:
            primary1 = np.array([
                [1.93869733e+00, 7.39498658e+02, 7.85710991e-01],
                [1.06429041e+00, 6.81360574e+02, 8.00262262e-01],
                [1.32884618e+00, 7.12760395e+02, 7.01095007e-01],
                ...
            ])

        The energy E can be given either in units of EeV or as log₁₀(E/eV).

    logE_start : float
        The logarithm (base 10) of the lowest event energy in eV.
        Example: if the lowest event energy is 1 EeV, then logE_start = 18.

    logE_end : float
        The logarithm (base 10) of the highest event energy in eV.


    ####################################################################
    Processed data format.
    
    Parameters
    ----------
    data : list of numpy.ndarray
        Each array corresponds to a primary and has shape (n_events, 4), with columns:
            1. weight
            2. Xmax [g/cm^2]
            3. dXmax [g/cm^2]
            4. log10(E/eV)
    
    Notes
    -----
    - Events are sorted by increasing log10(E/eV).
    - This format is compatible with the output of `DetectorEffects.include()`.
    """


 

    def __init__(self, data, logE_start, logE_end):
        self.table_efficiency = generate_table_xlambda()
        self.table_resolution = generate_table_sigmaf()
        self.logE_minimal = 17.8

        self.data = data
        self.num_primaries = len(data)
        self.logE_start = logE_start
        self.logE_end = logE_end
        
        self.get_logEbins()


        
    def get_logEbins(self):
        if self.logE_start < self.logE_minimal:
            raise ValueError("logE_start must be ≥ 17.8 to include detector effects.")
        if self.logE_start >= self.logE_end:
            raise ValueError("logE_start must be smaller than logE_end.")
    
        logE_lower  = max(np.round(self.logE_start, 1) - 0.1, self.logE_minimal)
        logE_higher = np.round(self.logE_end, 1) + 0.1
    
        self.logE_intervals = np.arange(logE_lower, logE_higher, 0.1)




        

        


    def include(self, data_unit):
        
        Y = np.arange(300, 1400+1)
        dY = Y[1] - Y[0]

        Simlong_events = []
        
        for z in range(self.num_primaries):
            print('Z = ' + str(z + 1))
            data_temp = self.data[z]
            
            
            if data_unit == 'EeV':
                logE = np.log10(data_temp[:, 0]) + 18
            elif data_unit == 'log10':
                logE = data_temp[:, 0]
            else:
                raise ValueError("data_unit must be either 'EeV' or 'log10' (log10(E/eV)).")

        
            logE__ = []
            pdf_X_n = []
            for idx_bin in range(len(self.logE_intervals)-1):
                logEmin = self.logE_intervals[idx_bin]
                logEmax = self.logE_intervals[idx_bin+1]
                key = f"{logEmin:.1f}_{logEmax:.1f}"
                print('logE_bin: ', key)
                x1, lam1, x2, lam2 = self.table_efficiency[key]
                sigma1, sigma2, f  = self.table_resolution[key]
                
                
                con = (logEmin <= logE) & (logE < logEmax)
                if con.sum() == 0:
                    continue
                
                xmax  = data_temp[con, 1] 
                dxmax = data_temp[con, 2] 
                logE__ += [logE[con]]
                # logE_bins are log_10(E/eV), while simulated data E unit is EeV
                
                
                #______________________I
                pdf_X_n_energy_bin = np.zeros((len(xmax), 3))
                for n in range(3):
                    detector_effects = non_central_moments(Y, sigma1 = sigma1, sigma2 = sigma2, f = f, n = n) * epsilon_fun(Y, x1 = x1, x2 = x2, lam1 = lam1, lam2 = lam2)
                    pdf_X_n_energy_bin[:,n] = np.array([ np.sum(normal_pdf(Y, mu = xmax[j], sigma = dxmax[j]) * detector_effects) * dY for j in range(len(xmax)) ])
                pdf_X_n += [pdf_X_n_energy_bin]
                
            pdf_X_n = np.vstack(pdf_X_n)
            logE__ = np.hstack(logE__)
            # OK
            
            #__________________________II
            weight_event = pdf_X_n[:,0]
            mean_event = pdf_X_n[:,1]/pdf_X_n[:,0]
            sigma_event = (pdf_X_n[:,2]/pdf_X_n[:,0] - mean_event**2)**0.5
        
            Simlong_events += [np.array([weight_event, mean_event, sigma_event, logE__]).T]
            # OK
            
        #______________________________III 
        # Simlong_events = np.array(Simlong_events, dtype = object)
        return Simlong_events 
        # OK



if __name__ == '__main__':
    """
    Example of using the DetectorEffects class.
    Make sure that `data_raw` is a list of numpy arrays in the correct format:
        Each array has shape (n_events, 3) with columns:
        [E [EeV], Xmax [g/cm^2], dXmax [g/cm^2]]
    """

    import os
    import numpy as np

    # File path
    data_dir = 'Simulated_data'
    filename = 'EPOS_xmax_Ebin2.npy'
    filepath = os.path.join(data_dir, filename)

    # Load data
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data_raw = np.load(filepath, allow_pickle=True)

    # Energy range (log10(E/eV)) ; 
    logE_start = 18
    logE_end   = 18.3

    # Initialize detector effects
    deteff = DetectorEffects(
                            data       = data_raw, 
                            logE_start = logE_start, 
                            logE_end   = logE_end
                            )

    # Include detector effects and get processed data
    data = deteff.include(data_unit='EeV')

       
    """
    Processed data format.
    
    Parameters
    ----------
    data : list of numpy.ndarray
        Each array corresponds to a primary and has shape (n_events, 4), with columns:
            1. weight
            2. Xmax [g/cm^2]
            3. dXmax [g/cm^2]
            4. log10(E/eV)
    
    Notes
    -----
    - Events are sorted by increasing log10(E/eV).
    - This format is compatible with the output of `DetectorEffects.include()`.
    """

    print("Data successfully processed. Example entry:")
    print(data[0][:5])  # show first 5 events of the first primary
    

