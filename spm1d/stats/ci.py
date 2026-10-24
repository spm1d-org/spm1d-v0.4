
'''
Confidence intervals

If the CI cloud starting from the "datum" reaches the "criterion", the null hypothesis is rejected.
'''

# Copyright (C) 2026  Todd Pataky


import numpy as np
from .. plot import plot_ci_0d, plot_ci_multisample_0d, plot_ci, plot_ci_multisample
from . _spm import df2str, dflist2str
from . t import ttest,ttest2






class _CI(object):
    ### summary attributes:
    name             = None    #confidence interval name ("0D CI" or "1D CI")
    dim              = None    #data dimensionality (0 or 1)
    kind             = None    #type of confidence interval (one-, paried- or two-sample)
    ### means, datum and criteria:
    criterion_type   = None    #criterion type (string)
    criterion_value  = None    #float or 1D array
    criterion        = None    #tuple: (criterion_type, criterion_value)
    datum_ci         = None    #primary ci ("ci" or "ciA")
    datum_type       = None    #datum type (string)
    datum_value      = None    #float or 1D array
    datum            = None    #tuple: (datum_type, datum_value)
    nMeans           = None    #number of means (1 or 2)
    mu               = None    #datum  (if applicable)
    ### probability:
    alpha            = None    #Type I error rate
    df               = None    #degrees of freedom
    zstar            = None    #critical test statistic value
    ### confidence interval:
    hstar            = None    #CI height
    h0reject         = None    #null hypothesis rejection decision
    ### booleans:
    ismultimean      = None    #is a multi-mean CI
    isparametric     = True    #is derived using parametric inference
    isscalarmu       = None    #is mu a scalar
    istest           = None    #is a hypothesis test

    def __init__(self, spmi):
        self.alpha           = spmi.alpha
        self.df              = spmi.df
        self.zstar           = spmi.zstar
        self.h0reject        = spmi.h0reject

    def __repr__(self):
        s            = self._get_repr_header()
        s           += self._get_repr_prob()
        s           += self._get_repr_ci()
        return s

    def _assemble_datum_criterion_pairs(self):
        self.datum       = self.datum_type, self.datum_value
        self.criterion   = self.criterion_type, self.criterion_value


    def _get_repr_header(self):
        s        = ''
        s       += '%s (%s)\n'                             %(self.name, self.get_percent(asstr=True))
        s       += '   kind          :  %s\n'              %self.kind
        s       += '   datum         :  %s  %s\n'          %(self.datum_type, self.get_datum_value(asstr=True, with_brackets=True))
        if self.istest:
            s   += '   criterion     :  %s  %s\n'          %(self.criterion_type, self.get_criterion_value(asstr=True, with_brackets=True))
        return s


    def _get_repr_prob(self):
        s            = ''
        s           += '   Probability\n'
        s           += '      alpha      :  %.3f\n'            %self.alpha
        if self.isparametric:
            s       += '      df         :  %s\n'              %dflist2str(self.df)
            if self.dim == 1:
                s   += '      fwhm       :  %.5f\n'            %self.fwhm
                s   += '      resels     :  %s\n'              %dflist2str(self.resels)
        s           += '      zstar      :  %.5f\n'            %self.zstar
        return s

    def _get_repr_ci(self):
        s        = ''
        s       += '   Confidence interval\n'
        s       += '      hstar      :  %s\n'              %self.get_hstar(asstr=True)
        if self.datum_type == 'mean':
            s   += '      ci         :  %s\n'              %self.get_ci(asstr=True)
        elif self.criterion_type == 'meanB':
            s   += '      ciA        :  %s\n'              %self.get_ciA(asstr=True)
        elif self.criterion_type == 'tailB':
            s   += '      ciA        :  %s\n'              %self.get_ciA(asstr=True)
            s   += '      ciB        :  %s\n'              %self.get_ciB(asstr=True)
        if self.istest:
            s   += '      h0reject   :  %s (%s)\n'         %(self.h0reject, self.get_h0rejection_decision_reason())
        return s

    def _get_ci(self, ci, asstr=False, with_brackets=False):
        x      = ci
        if asstr:
            if self.dim==0:
                x  = '(%.5f, %.5f)' %x
            else:
                x  = self.get_field_str(2, with_brackets)
        return x

    def _get_value(self, x, asstr=False, with_brackets=False):
        if isinstance(x, str):
            if with_brackets:
                x      = '(%s)' %x
        else:
            if asstr:
                if (self.dim==0) or (np.size(x)==1):
                    x  = '(%.5f)'%x if with_brackets else '%.5f'%x
                else:
                    x  = self.get_field_str(1, with_brackets)
        return x

    def get_ci(self, asstr=False, with_brackets=False):
        return self._get_ci(self.ci, asstr, with_brackets)
    def get_criterion_value(self, asstr=False, with_brackets=False):
        return self._get_value(self.criterion_value, asstr, with_brackets)
    def get_datum_value(self, asstr=False, with_brackets=False):
        return self._get_value(self.datum_value, asstr, with_brackets)
    def get_h0rejection_decision_reason(self):
        reason = 'outside' if self.h0reject else 'inside'
        return '%s %s %s' %(self.criterion_type, reason, self.datum_ci)
    def get_hstar(self, asstr=False, with_brackets=False):
        return self._get_value(self.hstar, asstr, with_brackets)
    def get_percent(self, asstr=False):
        x       = 100 * (1-self.alpha)
        if asstr:
            x   = '%.1f%s' %(x, '%')
        return x






class _CI0D(_CI):
    name             = '0D Confidence Interval'
    dim              = 0
    def plot(self, ax=None):
        plot_ci_0d(self, ax)


class _CI1D(_CI):
    name             = '1D Confidence Interval'
    dim              = 1
    Q                = None    #number of field nodes
    fwhm             = None    #field smoothness
    resels           = None    #resel counts

    def __init__(self, spmi):
        super(_CI1D, self).__init__(spmi)
        self.Q            = spmi.z.size
        self.fwhm         = spmi.fwhm
        self.resels       = spmi.resels

    def get_field_str(self, n=1, with_brackets=False):
        x            = '(%dx%d) field'  %(n, self.Q)
        if with_brackets:
            x        = '[%s]' %x
        return x

    def plot(self, ax=None, x=None, linecolor='k', facecolor='0.8', edgecolor='0.8', alpha=0.5, autoset_ylim=True):
        plot_ci(self, ax=ax, x=x, linecolor=linecolor, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, autoset_ylim=autoset_ylim)


def _asfloat(x):
    if isinstance(x, (float,int)):
        return float(x)
    elif isinstance(x, np.ndarray):
        return x.ravel()[0]


class _CISingleMean(_CI):
    kind             = 'One sample'
    nMeans           = 1
    ismultimean      = False
    mean             = None    #primary mean
    ci               = None    #confidence interval
    datum_ci         = 'ci'
    datum_type       = 'mean'
    criterion_type   = 'mu'

    def __init__(self, spmi, mean, hstar, mu=None):
        super(_CISingleMean, self).__init__(spmi)
        ### main parameters:
        self.mean            = mean if self.dim==1 else _asfloat(mean)
        self.mu              = mu
        ### confidence interval:
        self.hstar           = hstar
        self.ci              = self.mean - hstar, self.mean + hstar
        if self.dim == 1:
            self.ci          = np.asarray(self.ci)
        ### organizational attributes (redundant):
        self.isscalarmu      = isinstance(mu, (int,float))
        self.istest          = mu is not None
        self.datum_value     = self.mean
        self.criterion_value = self.mu
        self._assemble_datum_criterion_pairs()


class _CIDifference(_CISingleMean):
    kind             = 'Difference'




class _CIMultiMean(_CI):
    nMeans           = 2
    ismultimean      = True
    meanA            = None    #primary mean
    meanB            = None    #secondary mean
    ciA              = None    #primary confidence interval
    ciB              = None    #secondary confidence interval
    datum_ci         = 'ciA'
    datum_type       = 'meanA'

    def __init__(self, spmi, meanA, meanB, hstar, criterion='meanB'):
        super(_CIMultiMean, self).__init__(spmi)
        if criterion is None:
            criterion    = 'meanB'
        if criterion not in ('meanB', 'tailB'):
            raise ValueError('When datum is "meanA", "mu" must be None, "meanB" or "tailB". Received: %s' %repr(criterion))
        ### main parameters:
        self.meanA           = meanA if self.dim==1 else _asfloat(meanA)
        self.meanB           = meanB if self.dim==1 else _asfloat(meanB)
        ### confidence interval:
        self.hstar           = hstar
        if criterion == 'meanB':
            self.ciA         = self.meanA - self.hstar, self.meanA + self.hstar
            if self.dim == 1:
                self.ciA     = np.asarray(self.ciA)
        elif criterion == 'tailB':
            self.ciA         = self.meanA - 0.5*self.hstar, self.meanA + 0.5*self.hstar
            self.ciB         = self.meanB - 0.5*self.hstar, self.meanB + 0.5*self.hstar
            if self.dim == 1:
                self.ciA     = np.asarray(self.ciA)
                self.ciB     = np.asarray(self.ciB)
        ### organizational attributes (redundant):
        self.isscalarmu      = isinstance(self.meanA, (int,float))
        self.istest          = True
        self.datum_value     = self.meanA
        self.criterion_type  = criterion
        if criterion == 'meanB':
            self.criterion_value = self.meanB
        else:
            if self.dim == 0:
                self.criterion_value = self.ciB[0] if (self.meanA<self.meanB) else self.ciB[1]
            else:
                self.criterion_value = 'CI divergence'
        self._assemble_datum_criterion_pairs()


    def get_ci(self, asstr=False, with_brackets=False):
        return self.get_ciA(asstr, with_brackets)
    def get_ciA(self, asstr=False, with_brackets=False):
        return self._get_ci(self.ciA, asstr, with_brackets)
    def get_ciB(self, asstr=False, with_brackets=False):
        return self._get_ci(self.ciB, asstr, with_brackets)
    def plot(self, ax=None):
        plot_ci_multisample_0d(self, ax)



class _CIMultiMean1D(_CIMultiMean):
    def plot(self, ax=None, x=None, linecolor='k', facecolor='0.8', edgecolor='0.8', alpha=0.5, autoset_ylim=True):
        plot_ci(self, ax=ax, x=x, linecolor=linecolor, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, autoset_ylim=autoset_ylim)
class _CIMultiMean1DPlotter(object):
    def plot(self, ax=None, x=None, linecolors=('k','r'), facecolors=('0.8','r'), edgecolors=('0.4','r'), color_criterion='r', alphas=(0.5,0.5), autoset_ylim=True):
        plot_ci_multisample(self, ax=ax, x=x, linecolors=linecolors, facecolors=facecolors, edgecolors=edgecolors, color_criterion=color_criterion, alphas=alphas, autoset_ylim=autoset_ylim)




class _CIPairedSample(_CIMultiMean):
    kind             = 'Paired sample'
class _CITwoSample(_CIMultiMean):
    kind             = 'Two sample'
class _CIPairedDifference(_CIDifference):
    kind             = 'Paired difference'
class _CITwoSampleDifference(_CIDifference):
    kind             = 'Two-sample difference'




class CIOneSample0D(_CISingleMean, _CI0D):
    pass
class CIPairedSample0D(_CIPairedSample, _CI0D):
    pass
class CITwoSample0D(_CITwoSample, _CI0D):
    pass
class CIPairedDifference0D(_CIPairedDifference, _CI0D):
    pass
class CITwoSampleDifference0D(_CITwoSampleDifference, _CI0D):
    pass


class CIOneSample1D(_CISingleMean, _CI1D):
    pass
class CIPairedSample1D(_CIMultiMean1DPlotter, _CIPairedSample, _CI1D):
    pass
class CITwoSample1D(_CIMultiMean1DPlotter, _CITwoSample, _CI1D):
    pass
class CIPairedDifference1D(_CIPairedDifference, _CI1D):
    pass
class CITwoSampleDifference1D(_CITwoSampleDifference, _CI1D):
    pass








def ci_onesample(y, alpha=0.05, mu=None):
    spmi    = ttest(y, mu).inference(alpha, two_tailed=True)
    mean    = spmi.beta.flatten()  #sample mean
    mean    = mean if (mu is None) else (mean + mu)
    s       = spmi.sigma2**0.5     #sample standard deviation
    hstar   = spmi.zstar * s / y.shape[0]**0.5
    CIClass = CIOneSample1D if spmi.dim==1 else CIOneSample0D
    return CIClass(spmi, mean, hstar, mu)




def ci_pairedsample(yA, yB, alpha=0.05, datum='difference', mu=None):
    muvalue  = 0
    if datum=='difference':
        muvalue = 0 if mu is None else mu
    spmi     = ttest(yA-yB, muvalue).inference(alpha, two_tailed=True)
    mA,mB    = yA.mean(axis=0), yB.mean(axis=0)   #sample means
    s        = spmi.sigma2**0.5                   #sample standard deviation
    hstar    = spmi.zstar * s / yA.shape[0]**0.5  #CI height
    if datum=='difference':
        CIClass  = CIPairedDifference1D if spmi.dim==1 else CIPairedDifference0D
        ci       = CIClass(spmi, mA-mB, hstar, mu)
    else:
        CIClass  = CIPairedSample1D if spmi.dim==1 else CIPairedSample0D
        ci       = CIClass(spmi, mA, mB, hstar, mu)
    return ci




def ci_twosample(yA, yB, alpha=0.05, equal_var=True, datum='difference', mu=None):
    if equal_var is not True:
        raise NotImplementedError('Two-sample confidence interval calculations are currently implemented only for assumed equal variance. Set "equal_var=True" to force an equal variance assumption.')
    muvalue      = 0
    if datum=='difference':
        muvalue  = 0 if mu is None else mu
    spmi         = ttest2(yA-muvalue, yB, equal_var=True).inference(alpha, two_tailed=True)
    JA,JB        = yA.shape[0], yB.shape[0]
    mA,mB        = spmi.beta            #sample means
    mA          += muvalue
    s            = spmi.sigma2**0.5     #sample standard deviation
    hstar        = spmi.zstar * s * (1./JA + 1./JB)**0.5
    if datum == 'difference':
        CIClass  = CITwoSampleDifference1D if spmi.dim==1 else CITwoSampleDifference0D
        ci       = CIClass(spmi, mA-mB, hstar, mu)
    else:
        CIClass  = CITwoSample1D if spmi.dim==1 else CITwoSample0D
        ci       = CIClass(spmi, mA, mB, hstar, mu)
    return ci



