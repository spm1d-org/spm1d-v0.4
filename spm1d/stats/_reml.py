'''
Restricted maximum likelihood estimates

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)


The module contains functions which correct for non-sphericity.

The code is based on the Matlab routines in "spm_reml.m" as
released with SPM8 at www.fil.ion.ucl.ac.uk/spm/.

Specific version details:
    spm_reml.m
    $Id: spm_reml.m 1165 2008-02-22 12:45:16Z guillaume $
    authors: John Ashburner & Karl Friston
'''

# Copyright (C) 2026  Todd Pataky




import itertools
from copy import deepcopy
from math import sqrt,exp,log
import numpy as np



eps         = np.finfo(float).eps


def _rank(A, tol=None):
    '''
    This is a slight modification of np.linalg.matrix_rank.
    The tolerance performs poorly for some matrices
    Here the tolerance is boosted by a factor of ten for improved performance.
    '''
    M = np.asarray(A)
    S = np.linalg.svd(M, compute_uv=False)
    if tol is None:
        tol = 10 * S.max() * max(M.shape) * np.finfo(M.dtype).eps
    rank = sum(S > tol)
    return rank



def estimate_df_T(Y, X, eij, Q):
    n,s         = Y.shape
    ### ReML estimates:
    trRV        = n - _rank(X)
    ResSS       = (np.asarray(eij)**2).sum(axis=0)
    q           = np.diag(np.sqrt( trRV / ResSS )).T
    Ym          = Y @ q
    YY          = Ym@Ym.T / s
    V,h         = reml(YY, X, Q)
    V           = V * (n / np.trace(V))
    ### effective degrees of freedom:
    trRV,trRVRV = traceRV(V, X)
    df          = trRV**2 / trRVRV
    return df






def estimate_df_anova1(Y, X, eij, Q, c):
    n,s           = Y.shape                                #numbers ofresponses and nodes
    rankX         = _rank(X)
    ### ReML estimates:
    trRV          = n - rankX
    ResSS         = (np.asarray(eij)**2).sum(axis=0)
    q             = np.diag(np.sqrt( trRV / ResSS )).T
    Ym            = Y@q
    YY            = Ym@Ym.T / s
    V,h           = reml(YY, X, Q)
    V            *= (n / np.trace(V))
    ### effective degrees of freedom (denominator):
    trRV,trRVRV   = traceRV(V, X)
    df2           = trRV**2 / trRVRV
    ### effective degrees of freedom (numerator):
    # trMV,trMVMV   = traceMV(V, X, c)
    trMV,trMVMV   = traceMV(V, X)
    df1           =  trMV**2 / trMVMV
    df1           = max(df1,1.0)
    return df1, df2
    # df1,df2 = 2,2


def estimate_df_anova2(Y, X, eij, Q, C):
    J,s       = Y.shape
    rankX     = _rank(X)
    ### ReML estimates:
    trRV      = J - rankX
    ResSS     = (np.asarray(eij)**2).sum(axis=0)
    q         = np.diag(np.sqrt( trRV / ResSS )).T
    Ym        = Y@q
    YY        = Ym@Ym.T / s
    V,h       = reml(YY, X, Q)
    V        *= (J / np.trace(V))
    ### effective degrees of freedom (denominator):
    trRV,trRVRV = traceRV(V, X)
    df2         = trRV**2 / trRVRV
    ### effective degrees of freedom (numerator):
    df1         = []
    for CC in C:
        trMV,trMVMV = traceMV(V, X, CC)
        df1.append( trMV**2 / trMVMV )
    df1       = [max(x,1.0)  for x in df1]
    return df1, df2



# Translated from the SPM8 version of spm_reml.m:
# Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
# John Ashburner & Karl Friston
def reml(YY, X, Q, N=1, K=128):
    '''
    New version of reml  2025-05-18 which uses np.array instead of np.matrix
    to avoid NumPy linear algebra deprecation warnings
    '''
    n     = X.shape[0]
    W     = deepcopy(Q)
    q     = np.asarray(np.all(YY<np.inf, axis=0)).flatten()
    Q     = [QQ[q,:][:,q]   for QQ in Q]
    m     = len(Q)
    h     = np.array([float(np.any(np.diag(QQ)))  for QQ in Q]).T
    X0    = np.linalg.qr(X)[0]
    dFdhh = np.zeros((m,m))
    hE    = np.zeros(m)
    hP    = np.eye(m)/exp(32)
    dF    = np.inf
    for k in range(K):
        C    = np.zeros((n,n))
        for i in range(m):
            C   += Q[i] * float(h[i])
        iC   = np.linalg.inv(C) + np.eye(n)/exp(32)
        iCX  = iC @ X0
        Cq   = np.linalg.inv(X0.T@iCX)
        
        # Gradient dF/dh (first derivatives)
        P    = iC - iCX@Cq@iCX.T
        U    = np.eye(n) - P@YY/N
        PQ   = [P@QQ for QQ in Q]
        dFdh = np.array([-np.trace(PQQ@U)*0.5*N   for PQQ in PQ]).T
        
        # Expected curvature E{dF/dhh} (second derivatives)
        for i in range(m):
            for j in range(m):
                    dFdhh[i,j] = -np.trace(PQ[i]@PQ[j])*0.5*N
                    dFdhh[j,i] =  dFdhh[i,j]

        #add hyper-priors
        e      = h - hE
        dFdh  -= hP@e
        dFdhh -= hP
        
        # Fisher scoring
        dh    = -np.linalg.inv(dFdhh)@dFdh / log(k+3)
        h    += dh
        dF    = float(dFdh.T@dh)

        #final covariance estimate (with missing data points)
        if dF < 0.1:
            V     = 0
            for i in range(m):
                hh = h[i]
                hh = float(hh[0]) if isinstance(hh, np.ndarray) else float(hh)
                V += Q[i]*hh
            return V, h
    # maximum iterations reached:
    V     = 0
    for i in range(m):
        hh = h[i]
        hh = float(hh[0]) if isinstance(hh, np.ndarray) else float(hh)
        V += Q[i]*hh
    return V, h




# Translated from SPM12 spm_sp.m:
# Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
# Jean-Baptiste Poline
class MatrixWithProjections(object):  # following spm_sp:  Orthogonal (design) matrix space setting & manipulation
    def __init__(self, X):
        self.X   = X
        self.u   = None
        self.ds  = None
        self.v   = None
        self.tol = None
        self._svd()
        self.tol = max( X.shape ) * max(np.abs(self.ds)) * np.finfo(float).eps
        self.rk  = (self.ds > self.tol).sum()

    @property
    def m(self):
        return self.X.shape[0]
    @property
    def n(self):
        return self.X.shape[1]

    def _svd(self):
        from scipy.linalg import svd
        X  = self.X
        if X.shape[0] < X.shape[1]:
            v,s,u = svd(X.T, full_matrices=False, lapack_driver='gesvd')
        else:
            u,s,v = svd(X, full_matrices=False)
        self.u   = u
        self.ds  = s
        self.v   = v.T



# Translated from SPM12 spm_SpUtil.m:
# Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
# Andrew Holmes Jean-Baptiste Poline
def traceMV(V, X):  # a bit different from traceMV in _cov.py
    sX     = MatrixWithProjections( X )
    rk     = sX.rk
    u      = sX.u[:,:rk]
    Vu     = V @ u
    trMV   = (u.T * ( u.T @ V )).sum()
    trMVMV = np.linalg.norm( u.T @ Vu , ord='fro')**2
    return trMV, trMVMV



# Translated from SPM12 spm_SpUtil.m:
# Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
# Andrew Holmes Jean-Baptiste Poline
def traceRV(V, X):
    rank    = np.linalg.matrix_rank
    rk      = rank(X)
    sL      = X.shape[0]
    u       = np.linalg.qr(X)[0]
    Vu      = V @ u
    trV     = np.trace(V)
    trRVRV  = np.linalg.norm(V,  ord='fro')**2
    trRVRV -= 2*np.linalg.norm(Vu, ord='fro')**2
    trRVRV += np.linalg.norm(u.T @ Vu, ord='fro')**2
    trMV    = np.sum(  u * Vu  )  # element-wise multiplication (see spm8/spm_SpUtil.m, Line 550)
    trRV    = trV - trMV
    return trRV, trRVRV


