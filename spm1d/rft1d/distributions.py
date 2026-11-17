
'''
SciPy-like interface to 1D RFT distributions.
Currently implemented distributions include:

    * Gaussian
    * Student's t
    * Chi-squared
    * Fisher-Snedecor F
    * Hotelling's T2


All distributions share the following functions:

:Methods:

    **isf** --- RFT inverse survival function (critical height)

    **isf0d** --- Common (0D) inverse survival function (critical height)

    **p_cluster** --- RFT cluster-level inference

    **p_set** --- RFT set-level inference

    **sf** --- RFT survival function

    **sf0d** --- Common (0D) survival function.


:Basic use:

    All distributions can be accessed directly from **rft1d** as follows:

        >>> height = 3.0
        >>> nodes = 101
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)
        >>> rft1d.norm.sf0d(height)

:Unbroken and broken fields:

    If the field is unbroken (i.e. continuous) between the start and the
    end of the field, then the "nodes" argument to all methods should be
    an integer representing the number of field nodes:

        >>> height = 3.0
        >>> nodes = 101
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)

    However, if the field is broken (i.e. piecewise continuous), then the
    "nodes" argument should be a 1D mask:  a boolean array (size: nodes)
    where False specifies nodes which are masked out:

        >>> height = 3.0
        >>> nodes = np.array([True]*20 + [False]*30 + [True]*51)
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)


:Very rough fields and the Bonferroni correction:

    When the fields are very rough (e.g. *FWHM* < 2),
    the Bonferroni correction may be less severe than the RFT correction.
    While both are vaild, it is generally best to use the less-severe
    threshold to maintain statistical power.
    To adopt the less-severe threshold use the keyword argument
    *withBonf* as follows:

        >>> rft1d.norm.sf(3, 101, 1.5, withBonf=False) #yields 0.179
        >>> rft1d.norm.sf(3, 101, 1.5, withBonf=True)  #yields 0.136

    By default the *withBonf* argument is False.

        >>> rft1d.norm.sf(3, 101, 1.5)  #yields 0.179

    For smooth fields the keyword argument will have no effect:

        >>> rft1d.norm.sf(3, 101, 5.0, withBonf=False) #yields 0.0590
        >>> rft1d.norm.sf(3, 101, 5.0, withBonf=True)  #yields 0.0590


:Very smooth fields:

    When Gaussian fields become very smooth, they start to behave like
    Gaussian scalars. Theoretically RFT results are equivalent to scalar
    results when smoothness is infinite. Thus raising the *FWHM* value
    systematically will cause the RFT results to converge to typical 0D results:

        >>> scipy.stats.norm.sf(2)  #yields 0.02275
        >>> rft1d.norm.sf(3, 101, 10.0) #yields 0.31710
        >>> rft1d.norm.sf(3, 101, 100.0) #yields 0.05693
        >>> rft1d.norm.sf(3, 101, 1000.0) #yields 0.02599
        >>> rft1d.norm.sf(3, 101, 10000.0) #yields 0.02284
        >>> rft1d.norm.sf(3, 101, 100000.0) #yields 0.02275

    Setting the smoothness to infinite will return the same result:

        >>> rft1d.norm.sf(3, 101, np.inf) #yields 0.02275

'''

# Copyright (C) 2026  Todd Pataky


import types
import numpy as np
from scipy import stats
from . prob import RFTCalculator, RFTCalculatorResels


_DF_EXAMPLES = {
	0 : (('DOFFLAG2', ''), ('DOFFLAG', ''),
	     ('\n\t\t\t*df* -- degrees of freedom (int or float)\n', '')),
	1 : (('DOFFLAG3', ', 8'), ('DOFFLAG2', ', 8'), ('DOFFLAG', ' 8,')),
	2 : (('DOFFLAG3', ', 4, 21'), ('DOFFLAG2', ', (3,15)'), ('DOFFLAG', ' (2,14),'),
	     ('degrees of freedom (int or float)', 'degrees of freedom (two-tuple of int or float)')),
}


def _fill_docstring(doc, distname, ndf):
	'''Substitute the distribution name and the degree-of-freedom examples.'''
	doc = doc.replace('DISTFLAG', distname)
	for old, new in _DF_EXAMPLES.get(ndf, ()):
		doc = doc.replace(old, new)
	return doc


def _base_docstring(cls, name):
	'''Docstring of method *name* from the first direct base that has one.'''
	for base in cls.__bases__:
		doc = getattr(getattr(base, name, None), '__doc__', None)
		if doc:
			return doc
	return None


def add_docstrings(distname, ndf=0):
	def decorate(cls):
		undocumented = [(name, attr)  for name, attr in vars(cls).items()
		                if isinstance(attr, types.FunctionType) and not attr.__doc__]
		for name, method in undocumented:
			doc = _base_docstring(cls, name)
			if doc is not None:
				method.__doc__ = _fill_docstring(doc, distname, ndf)
		return cls
	return decorate


class _RFTDistribution(object):
	def __init__(self, STAT, ndf):
		self._STAT  = STAT
		self._ndf   = int(ndf)
	def _get_df(self, df):
		if self._ndf==1:
			df = 1, df
		return df
	def isf(self, alpha, df, nodes, FWHM, withBonf=False):
		'''
		RFT inverse survival function.
		(see also the survival function: **rft1d.DISTFLAG.sf**)

		:Parameters:

			*alpha* -- upper tail probability (float;  0 < alpha < 1)

			*df* -- degrees of freedom (int or float)
			
			*nodes* -- number of field nodes (int)  OR a binary field (boolean array)

			*FWHM* -- field smoothness (float)

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

		:Returns:

			Quantile corresponding to upper-tail probability alpha.
			Equivalently: critical threshold at a Type I error rate of alpha.

		:Examples:

			>>> rft1d.DISTFLAG.isf(0.05,DOFFLAG 101, 10.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculator(STAT=self._STAT, df=df, nodes=nodes, FWHM=FWHM, withBonf=withBonf)
		return E.isf( alpha )

	def isf_resels(self, alpha, df, resels, withBonf=False, nNodes=None):
		'''
		RFT inverse survival function.
		(see also the survival function: **rft1d.DISTFLAG.sf**)

		:Parameters:

			*alpha* -- upper tail probability (float;  0 < alpha < 1)

			*df* -- degrees of freedom (int or float)
			
			*resels* -- resolution element counts

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)
			
			*nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)

		:Returns:

			Quantile corresponding to upper-tail probability alpha.
			Equivalently: critical threshold at a Type I error rate of alpha.

		:Examples:

			>>> rft1d.DISTFLAG.isf(0.05,DOFFLAG 101, 10.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculatorResels(STAT=self._STAT, df=df, resels=resels, withBonf=withBonf, nNodes=nNodes)
		return E.isf( alpha )
	def isf0d(self):
		'''
		Inverse survival function (0D);  equivalent to **scipy.stats.DISTFLAG.isf**
		
		:Examples:

			>>> rft1d.DISTFLAG.isf0d([0.01, 0.05, 0.1]DOFFLAG2)
			>>> scipy.stats.DISTFLAG.isf([0.01, 0.05, 0.1]DOFFLAG3)
		'''
		pass
	def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
		'''
		RFT cluster-level inference.
		
		Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
		an upcrossing of extent *k* when thresholded at *u*.
		For set-specific probabilities use **rft1d.DISTFLAG.p_set**
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**
		
		:Parameters:

			*k* -- cluster extent (resels)
			
			*u* -- threshold

			*df* -- degrees of freedom (int or float)
			
			*nodes* -- number of field nodes (int)  OR a binary field (boolean array)

			*FWHM* -- field smoothness (float)

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

		:Returns:

			Cluster-specific probability value.

		:Examples:

			>>> rft1d.DISTFLAG.p_cluster(0.5, 3.0,DOFFLAG 101, 15.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculator(STAT=self._STAT, df=df, nodes=nodes, FWHM=FWHM, withBonf=withBonf)
		return E.p.cluster(k, u)

	def p_cluster_resels(self, k, u, df, resels, withBonf=False, nNodes=None):
		'''
		RFT cluster-level inference.
		
		Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
		an upcrossing of extent *k* when thresholded at *u*.
		For set-specific probabilities use **rft1d.DISTFLAG.p_set**
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**
		
		:Parameters:

			*k* -- cluster extent (resels)
			
			*u* -- threshold

			*df* -- degrees of freedom (int or float)
			
			*resels* -- resolution element counts

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

			*nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)

		:Returns:

			Cluster-specific probability value.

		:Examples:

			>>> rft1d.DISTFLAG.p_cluster(0.5, 3.0,DOFFLAG 101, 15.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculatorResels(STAT=self._STAT, df=df, resels=resels, withBonf=withBonf, nNodes=nNodes)
		return E.p.cluster(k, u)

	def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
		'''
		RFT set-level inference.
		
		Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
		at least *c* upcrossings with a minimum extent of *k* when thresholded at *u*.
		This probability pertains to the entire excursion set.
		For cluster-specific probabilities use **rft1d.DISTFLAG.p_cluster**
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**
		
		:Parameters:

			*c* -- number of upcrossings
			
			*k* -- minimum cluster extent (resels)
			
			*u* -- threshold

			*df* -- degrees of freedom (int or float)
			
			*nodes* -- number of field nodes (int)  OR a binary field (boolean array)

			*FWHM* -- field smoothness (float)

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

		:Returns:

			Set-specific probability value.

		:Examples:

			>>> rft1d.DISTFLAG.p_set(2, 0.5, 3.0,DOFFLAG 101, 15.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculator(STAT=self._STAT, df=df, nodes=nodes, FWHM=FWHM, withBonf=withBonf)
		return E.p.set(c, k, u)
		
	def p_set_resels(self, c, k, u, df, resels, withBonf=False, nNodes=None):
		'''
		RFT set-level inference.
		
		Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
		at least *c* upcrossings with a minimum extent of *k* when thresholded at *u*.
		This probability pertains to the entire excursion set.
		For cluster-specific probabilities use **rft1d.DISTFLAG.p_cluster**
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**
		
		:Parameters:

			*c* -- number of upcrossings
			
			*k* -- minimum cluster extent (resels)
			
			*u* -- threshold

			*df* -- degrees of freedom (int or float)
			
			*resels* -- resolution element counts

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

			*nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)

		:Returns:

			Set-specific probability value.

		:Examples:

			>>> rft1d.DISTFLAG.p_set(2, 0.5, 3.0,DOFFLAG 101, 15.0)
		'''
		df = self._get_df(df)
		E  = RFTCalculatorResels(STAT=self._STAT, df=df, resels=resels, withBonf=withBonf, nNodes=nNodes)
		return E.p.set(c, k, u)
	
	def sf(self, u, df, nodes, FWHM, withBonf=False):
		'''
		RFT survival function.
		
		Probability that 1D Gaussian fields with a smoothness *FWHM* would produce a 1D statistic field whose maximum exceeds *u*.

		:Parameters:

			*u* -- threshold (int, float, or sequence of int or float)

			*df* -- degrees of freedom (int or float)
			
			*nodes* -- number of field nodes (int)  OR a binary field (boolean array)

			*FWHM* -- field smoothness (float)

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

		:Returns:

			The probability of exceeding the specified heights.

		:Examples:

			>>> rft1d.DISTFLAG.sf([1,2,3,4,5],DOFFLAG 101, 10.0)
		'''
		df   = self._get_df(df)
		calc = RFTCalculator(STAT=self._STAT, df=df, nodes=nodes, FWHM=FWHM, withBonf=withBonf)
		return calc.sf( u )

	def sf_resels(self, u, df, resels, withBonf=False, nNodes=None):
		'''
		RFT survival function.
		
		Probability that 1D Gaussian fields with a smoothness *FWHM* would produce a 1D statistic field whose maximum exceeds *u*.

		:Parameters:

			*u* -- threshold (int, float, or sequence of int or float)

			*df* -- degrees of freedom (int or float)
			
			*resels* -- resolution element counts

			*withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)

			*nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)

		:Returns:

			The probability of exceeding the specified heights.

		:Examples:

			>>> rft1d.DISTFLAG.sf([1,2,3,4,5],DOFFLAG 101, 10.0)
		'''
		df   = self._get_df(df)
		calc = RFTCalculatorResels(STAT=self._STAT, df=df, resels=resels, withBonf=withBonf, nNodes=nNodes)
		return calc.sf( u )


	def sf0d(self):
		'''
		Survival function (0D);  equivalent to **scipy.stats.DISTFLAG.sf**
		
		:Examples:

			>>> rft1d.DISTFLAG.sf0d([0,1,2]DOFFLAG2)
			>>> scipy.stats.DISTFLAG.sf([0,1,2]DOFFLAG3)
		'''
		pass


@add_docstrings('norm', ndf=0)
class Gaussian(_RFTDistribution):
	'''
	Gaussian distributions are accessible via **rft1d.norm** and **rft1d.distributions.norm**
	'''
	def __init__(self):
		super(Gaussian, self).__init__('Z', 0)
	def isf(self, alpha, nodes, FWHM, withBonf=False):
		return super(Gaussian, self).isf(alpha, None, nodes, FWHM, withBonf)
	def isf0d(self, alpha):	
		return stats.norm.isf(alpha)
	def p_cluster(self, k, u, nodes, FWHM, withBonf=False):
		return super(Gaussian, self).p_cluster(k, u, None, nodes, FWHM, withBonf)
	def p_set(self, c, k, u, nodes, FWHM, withBonf=False):
		return super(Gaussian, self).p_set(c, k, u, None, nodes, FWHM, withBonf)
	def sf(self, u, nodes, FWHM, withBonf=False):
		return super(Gaussian, self).sf(u, None, nodes, FWHM, withBonf)
	def sf0d(self, heights):
		return stats.norm.sf(heights)


@add_docstrings('t', ndf=1)
class StudentsT(_RFTDistribution):
	def __init__(self):
		super(StudentsT, self).__init__('T', 1)
	def isf(self, alpha, df, nodes, FWHM, withBonf=False):
		return super(StudentsT, self).isf(alpha, df, nodes, FWHM, withBonf)
	def isf0d(self, alpha, df):	
		return stats.t.isf(alpha, df)
	def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
		return super(StudentsT, self).p_cluster(k, u, df, nodes, FWHM, withBonf)
	def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
		return super(StudentsT, self).p_set(c, k, u, df, nodes, FWHM, withBonf)
	def sf(self, u, df, nodes, FWHM, withBonf=False):
		return super(StudentsT, self).sf(u, df, nodes, FWHM, withBonf)
	def sf0d(self, u, df):
		return stats.t.sf(u, df)


@add_docstrings('chi2', ndf=1)
class Chi2(_RFTDistribution):
	def __init__(self):
		super(Chi2, self).__init__('X2', 1)
	def isf(self, alpha, df, nodes, FWHM, withBonf=False):
		return super(Chi2, self).isf(alpha, df, nodes, FWHM, withBonf)
	def isf0d(self, alpha, df):	
		return stats.chi2.isf(alpha, df)
	def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
		return super(Chi2, self).p_cluster(k, u, df, nodes, FWHM, withBonf)
	def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
		return super(Chi2, self).p_set(c, k, u, df, nodes, FWHM, withBonf)
	def sf(self, u, df, nodes, FWHM, withBonf=False):
		return super(Chi2, self).sf(u, df, nodes, FWHM, withBonf)
	def sf0d(self, u, df):
		return stats.chi2.sf(u, df)


@add_docstrings('f', ndf=2)
class FisherSnedecorF(_RFTDistribution):
	def __init__(self):
		super(FisherSnedecorF, self).__init__('F', 2)
	def isf(self, alpha, df, nodes, FWHM, withBonf=False):
		return super(FisherSnedecorF, self).isf(alpha, df, nodes, FWHM, withBonf)
	def isf0d(self, alpha, df):	
		return stats.f.isf(alpha, df[0], df[1])
	def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
		return super(FisherSnedecorF, self).p_cluster(k, u, df, nodes, FWHM, withBonf)
	def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
		return super(FisherSnedecorF, self).p_set(c, k, u, df, nodes, FWHM, withBonf)
	def sf(self, u, df, nodes, FWHM, withBonf=False):
		return super(FisherSnedecorF, self).sf(u, df, nodes, FWHM, withBonf)
	def sf0d(self, u, df):
		return stats.f.sf(u, df[0], df[1])


@add_docstrings('T2', ndf=2)
class HotellingsT2(_RFTDistribution):
	def __init__(self):
		super(HotellingsT2, self).__init__('T2', 2)
	def isf(self, alpha, df, nodes, FWHM, withBonf=False):
		return super(HotellingsT2, self).isf(alpha, df, nodes, FWHM, withBonf)
	def isf0d(self, alpha, df):
		'''
		0D inverse survival function for the Hotelling's T2 distribution
		(not implemented in **scipy.stats**)
		
		:Examples:

			>>> rft1d.T2.isf0d(0.05, (2,14))
			>>> rft1d.T2.isf0d([0.01, 0.05, 0.10], (2,14))
		'''
		p,m    = map(float,df)
		df_F   = p, m - p + 1
		fstar  = stats.f.isf(alpha, df_F[0], df_F[1])
		T2star = fstar / ( (m-p+1)/(p*m) )
		return T2star
	def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
		return super(HotellingsT2, self).p_cluster(k, u, df, nodes, FWHM, withBonf)
	def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
		return super(HotellingsT2, self).p_set(c, k, u, df, nodes, FWHM, withBonf)
	def sf(self, u, df, nodes, FWHM, withBonf=False):
		return super(HotellingsT2, self).sf(u, df, nodes, FWHM, withBonf)
	def sf0d(self, u, df):
		'''
		0D survival function for the Hotelling's T2 distribution
		(not implemented in **scipy.stats**)
		
		:Examples:

			>>> rft1d.T2.sf0d([0,1,2], (2,14))
		'''
		p,m    = map(float,df)
		if isinstance(u, (int,float)):
			uF = u * ( (m-p+1)/(p*m) )
		else:
			uF = np.asarray(u) * ( (m-p+1)/(p*m) )
		v1,v2  = p, m - p + 1
		return stats.f.sf(uF, v1, v2)





norm   = Gaussian()
t      = StudentsT()
chi2   = Chi2()
f      = FisherSnedecorF()
T2     = HotellingsT2()




