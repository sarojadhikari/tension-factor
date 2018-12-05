import numpy as np

from likelihoods import Planck_plik_lite_likelihood

class split_likelihood(Planck_plik_lite_likelihood):
    def __init__(self, which="TTsplit", lsplit=1000):
        #super().__init__(which)
        self.tausigma = np.inf
        self.lsplit = lsplit
        self.bsplit = 114 # get the nearest bin for the split; currently hard set for 1000
        super().__init__(which, tausigma=self.tausigma)

    def set_data_and_covariance(self):
        if (self.which == "TTsplit"):
            self.cldata = self.clTT
            self.cov = self.gcovTT
        elif (self.which == "aTTsplit"):
            self.cldata = self.clTT[self.bsplit:]
            self.cov = self.gcovTT[self.bsplit:, self.bsplit:]
            self.bmaTT = self.bmTT[self.bsplit:, self.lsplit-30:2509-30]
        elif (self.which == "bTTsplit"):
            self.cldata = self.clTT[0:self.bsplit]
            self.cov = self.gcovTT[0:self.bsplit, 0:self.bsplit]
            self.bmbTT = self.bmTT[0:self.bsplit, 0:self.lsplit-30]
        elif (self.which == "EEsplit"):
            self.cldata = self.clEE
            self.cov = self.gcovEE
        elif (self.which == "aEEsplit"):
            self.cldata = self.clEE[self.bsplit:]
            self.cov = self.gcovEE[self.bsplit:, self.bsplit:]
            self.bmaEE = self.bmEE[self.bsplit:, self.lsplit-30:1997-30]
        elif (self.which == "bEEsplit"):
            self.cldata = self.clEE[0:self.bsplit]
            self.cov = self.gcovEE[0:self.bsplit, 0:self.bsplit]
            self.bmbEE = self.bmEE[0:self.bsplit, 0:self.lsplit-30]

        self.invcov = np.linalg.inv(self.cov)

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
        if (self.which == "TTsplit"):
            clthb = self.bmTT@(self.mufac*(self.cmb.cambTCls[30:2509]))
        elif (self.which == "EEsplit"):
            clthb = self.bmEE@(self.mufac*(self.cmb.cambECls[30:1997]))
        elif (self.which == "aTTsplit"):
            clthb = self.bmaTT@(self.mufac*(self.cmb.cambTCls[self.lsplit:2509]))
        elif (self.which == "bTTsplit"):
            clthb = self.bmbTT@(self.mufac*(self.cmb.cambTCls[30:self.lsplit]))
        elif (self.which == "aEEsplit"):
            clthb = self.bmaEE@(self.mufac*(self.cmb.cambECls[self.lsplit:1997]))
        elif (self.which == "bEEsplit"):
            clthb = self.bmbEE@(self.mufac*(self.cmb.cambECls[30:self.lsplit]))
        elif (self.which == "TTEE"):
            clthbT = self.bmTT@(self.mufac*(self.cmb.cambTCls[30:2509]))
            clthbE = self.bmEE@(self.mufac*(self.cmb.cambECls[30:1997]))
            clthb = np.append(clthbT, clthbE)
        
        cldif = self.cldata - clthb
        
        return -0.5*(cldif@self.invcov@cldif) + tauprior
        
    def neglogLike(self, params):
    # negative of loglike -- useful for minimizing
        return (-1.)*self.logLike(params)
    
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
