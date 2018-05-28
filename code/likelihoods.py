import numpy as np
from cosmology.cmb import CMB
from plik_binning import binning_matrix

bin_matrix = binning_matrix()

class Planck_plik_lite_likelihood(object):
    """
        instead of using the planck clik code, we will use the cl and
        covariance data provided by Planck for the plik_lite likelihood
        and implement the multivariate Gaussian log-likelihood
    """
    
    def __init__(self, which="TT", taumean=0.07, tausigma=0.02):
        # initialize 
        self.taumean = taumean
        self.tausigma = tausigma
        self.which = which
        
        self.bounds = [[2.7, 3.4],      # log10As
                       [0.8, 1.2],      # ns
                       [50, 95],        # H0
                       [0.1, 0.45],     # Om
                       [0.044, 0.056],  # Ob
                       [0.005, 0.2]]    # tau
        
        self.mufac = (2.7255E6)**2.0 # conversion factor to muK^2

        self.read_data()
        self.initialize_camb()   
        self.set_data_and_covariance()
        
    def read_data(self):
        # read plik_lite cl and covariance data from plik_lite_data/
        cldata = np.loadtxt('plik_lite_data/cl_cmb_plik_v18.dat')
        # these data are Cls (not Dls) in mu K^2
        lbins = 613; lbinsTT = 215; lbinsEE = 199
        lmin = 30; lmaxTT = 2508; lmaxEE = 1996
        
        self.clTCE = cldata[0:lbins, 1]
        self.clTT = self.clTCE[0:lbinsTT]
        self.clEE = self.clTCE[lbinsTT+lbinsEE:]

        self.leff = np.array([int(cldata[i,0]) for i in range(0, 613)])
        self.gcov = np.load("plik_lite_data/c_matrix_plik_v18.npy")
        self.gcovTT = self.gcov[0:lbinsTT, 0:lbinsTT]
        self.gcovEE = self.gcov[lbinsTT+lbinsEE:, lbinsTT+lbinsEE:]
        self.gcovTE = self.gcov[0:lbinsTT, lbinsTT+lbinsEE:]
        self.gcovET = self.gcov[lbinsTT+lbinsEE:, 0:lbinsTT]
        
        self.gcovTTEE = np.bmat([[self.gcovTT, self.gcovTE],
                                 [self.gcovET, self.gcovEE]])
        
        # also generate various binning matrices
        self.bmTT = bin_matrix[0:lbinsTT, 0:lmaxTT-lmin+1]
        self.bmEE = bin_matrix[lbinsTT+lbinsEE:, lmaxTT+lmaxEE-60+2:]
        
    def initialize_camb(self, pol=1):
        self.cmb = CMB(camb_init=False)
        lmax = 2508
        self.cmb.init_camb(aboost=1, lboost=1, LMAX=lmax)
        self.cmb.cambparams.AccuratePolarization = pol
        self.cmb.cambparams.DoLensing = 1
    
    def set_data_and_covariance(self):
        if (self.which == "TT"):
            self.cldata = self.clTT
            self.cov = self.gcovTT
        elif (self.which == "EE"):
            self.cldata = self.clEE
            self.cov = self.gcovEE
        elif (self.which == "TTEE"):
            self.cldata = np.append(self.clTT, self.clEE)
            self.cov = self.gcovTTEE
        
        self.invcov = np.linalg.inv(self.cov)
        
    def get_camb_Cls(self, params):
        logA, ns, H0, Oc, Ob, tau = params
        
        self.cmb.cosmology.set_H0(H0)
        self.cmb.cosmology.set_n(ns)
        self.cmb.cosmology.set_A((9./25)*np.exp(logA)*1.e-10)
        self.cmb.cosmology.set_Oc0(Oc)
        self.cmb.cosmology.set_Ob0(Ob)
        self.cmb.cosmology.set_tau(tau)
        
        self.cmb.set_camb_cosmology()
        self.cmb.get_camb_results()
        
    def logLike(self, params):
        # evaluate the loglikelihood including the tau prior
        
        # first check if any of the parameters are out-of-bounds
        for i in range(len(params)):
            if (params[i]>self.bounds[i][1]) or (params[i]<self.bounds[i][0]):
                return -np.infty
            
        self.get_camb_Cls(params)
        tau = params[5]
        tauprior = -0.5*np.power((tau-self.taumean)/self.tausigma, 2.0)
        
        # first obtain the binned theory cls
        if (self.which == "TT"):
            clthb = self.bmTT@(self.mufac*(self.cmb.cambTCls[30:2509]))
        elif (self.which == "EE"):
            clthb = self.bmEE@(self.mufac*(self.cmb.cambECls[30:1997]))
        elif (self.which == "TTEE"):
            clthbT = self.bmTT@(self.mufac*(self.cmb.cambTCls[30:2509]))
            clthbE = self.bmEE@(self.mufac*(self.cmb.cambECls[30:1997]))
            clthb = np.append(clthbT, clthbE)
        
        cldif = self.cldata - clthb
        
        return -0.5*(cldif@self.invcov@cldif) + tauprior
    
### sample code to get the best-fit cosmological parameters using scipy ###
###########################################################################
        
    """
    from likelihoods import Planck_plik_lite_likelihood
    plikliteTT = Planck_plik_lite_likelihood(which="TT")
    
    def lnLikeT(params):
        return (-1.)*plikliteTT.logLike(params)
        
    from scipy.optimize import differential_evolution as dev
    bounds = plikliteTT.bounds
    res = dev(lnLikeT, bounds, disp = True)
    
    """