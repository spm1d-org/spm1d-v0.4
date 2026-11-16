
'''
Random Field Theory expectations and probabilities.
The core RFT computations are conducted inside **prob.rft**, and the 
**RFTCalculator** class serves as a high-level interface to **prob.rft**
'''

# Copyright (C) 2026  Todd Pataky



from math import pi,log,sqrt,exp
import numpy as np
from scipy import stats,optimize
from scipy.special import gammaln,gamma
from . import geom

# CONSTANTS:
FOUR_LOG2   = 4*log(2)
SQRT_4LOG2  = sqrt(4*log(2))
SQRT_2      = sqrt(2)
TWO_PI      = 2*pi
# eps         = np.finfo(np.float).eps
eps         = np.finfo(float).eps



def p_bonferroni(STAT, z, df, Q, n=1):
	'''
	Bonferroni correction.
	
	When fields are very rough a Bonferroni correction might be less severe than
	the RFT threshold. This function yields Bonferroni-corrected p values based
	on the number of field nodes *Q*.
	
	:Parameters:

		*STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")
		
		*z* --- field height
		
		*df* --- degrees of freedom [df{interest} df{error}]
		
		*Q* --- number of field nodes (used for Bonferroni comparison)

		*n* --- number of test statistic fields in conjunction

	:Returns:

		The probability of exceeding the specified height.

	:Example:

		>>> rft1d.prob.p_bonferroni('Z', 3.1, None, 101) #yields 0.098
	
	'''
	if STAT=='Z':
		p     = stats.norm.sf(z)
	if STAT=='T':
		p     = stats.t.sf(z, df[1])
	elif STAT=='F':
		p     = stats.f.sf(z, df[0], df[1])
	elif STAT=='X2':
		p     = stats.chi2.sf(z, df[1])
	elif STAT=='T2':
		p,m   = map(float,df)
		v0,v1 = p, m - p + 1
		zz    = z * ( (m-p+1)/(p*m) )
		p     = stats.f.sf(zz, v0, v1)
	p         = Q * (p**n)
	return min(p, 1)


def _replaceWithBonferroniIfPossible(STAT, P, c, csize, z, df, Q, n=1):
	if (csize is None) or (z is None) or (Q is None) or (n is None):
		return P
	if (c==1) & (csize==0) :
		Pbonf  = p_bonferroni(STAT, z, df, Q, n)
		P      =  min(P, Pbonf)
	return P

def _replaceWith0DpValueIfPossible(STAT, P, c, csize, z, df, Q, n=1):
	if (c>1) or (csize>0):
		return P
	p = p_bonferroni(STAT, z, df, 1, n)
	if p>P:
		return p
	else:
		return P

# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_Z(z):
	ec0d        = 1 - stats.norm.cdf(z)
	ec1d        = SQRT_4LOG2 / TWO_PI   *  exp(-0.5*(z*z))
	return [ec0d, ec1d]


# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_T(z, df):
	'''
	Reference:  Worsley KJ et al. (1996) Hum Brain Mapp 4:58-73
	Reference:  Worsley KJ et al. (2004) [Eqn.2 and Table 2]
	'''
	v    = float(df[1])
	a    = FOUR_LOG2
	b    = np.exp((gammaln((v+1)/2) - gammaln(v/2)))
	c    = (1+z**2/v)**((1-v)/2)
	EC   = []
	EC.append(  1 - stats.t.cdf(z,v)  )  #dim: 0
	EC.append(  a**0.5 / TWO_PI * c  )   #dim: 1
	return EC

# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_F(z, df):
	if z<0:   
		return [1, np.inf]    #to bypass warnings in critical threshold calculation
	k,v  = map(float, df)
	k    = max(k, 1.0)        #stats.f.cdf will return nan if k is less than 1
	a    = FOUR_LOG2/TWO_PI
	b    = gammaln(v/2) + gammaln(k/2)
	EC   = []
	EC.append(  1 - stats.f.cdf(z, k, v)  )
	EC.append(  a**0.5 * np.exp(gammaln((v+k-1)/2)-b)*2**0.5 *(k*z/v)**(0.5*(k-1))*(1+k*z/v)**(-0.5*(v+k-2))  )
	return EC

# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_X2(z, df):
	v    = float(df[1])
	a    = FOUR_LOG2 / TWO_PI
	b    = z ** ((v-1)/2)  * np.exp(-z/2 -gammaln(v/2))  /  (2**((v-2)/2))
	EC   = []
	EC.append(  1 - stats.chi2.cdf(z,v)  )
	EC.append(  a**0.5 * b  )
	return EC


def ec_density(STAT, z, df):
	if STAT=='Z':
		return ec_density_Z(z)
	if STAT=='T':
		return ec_density_T(z, df)
	elif STAT=='F':
		return ec_density_F(z, df)
	elif STAT=='X2':
		return ec_density_X2(z, df)
	elif STAT=='T2':
		p,m  = map(float,df)
		df_F = p, m - p + 1
		zz   = z * ( (m-p+1)/(p*m) )
		return ec_density_F(zz, df_F)
	else:
		raise(ValueError('Statistic must be one of: ["Z", "T", "X2", "F", "T2"]'))


def poisson_cdf(a, b):
	# return stats.poisson.cdf(a, b)
	# returns zero when b<0 to matches spm8 results
	if b <= 0:
		p   = 0.0
	else:
		p   = stats.poisson.cdf(a, b)
	return p


# Translated from SPM12 spm_P_RF.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def rft(c, k, STAT, Z, df, R, n=1, Q=None, expectations_only=False, version='spm12'):
	'''
	Random Field Theory probabilities and expectations using unified Euler Characteristic (EC) theory.
	This code is based on "spm_P_RF.m" and "spm_P.m" from the spm8 and spm12 Matlab packages
	which are available from: http://www.fil.ion.ucl.ac.uk/spm/
	
	:Parameters:
	
		*c* --- number of clusters
		
		*k* --- cluster extent (resels)
		
		*STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")
		
		*Z* --- field height
		
		*df* --- degrees of freedom [df{interest} df{error}]
		
		*R* --- resel counts (0D counts, 1D counts) defining search volume
		
		*n* --- number of test statistic fields in conjunction
		
		*Q* --- number of field nodes (used for Bonferroni comparison)
		
		*expectations_only* --- if True only expectations will be returned
		
		*version* --- "spm8" or "spm12" (see below)
	
	:Returns:
	
		*P*  --- corrected P value
		
		*p*  --- uncorrected P value
		
		*Ec* --- expected number of upcrossings {c}
		
		*Ek* --- expected resels per upcrossing {k}
		
		*EN* --- expected excursion set resels
		
		NOTE!  If expectations_only==True, then only (Ec,Ek,EN) are returned.
	
	:Examples:
	
		>>> P,p,Ec,Ek,EN = rft1d.prob.rft(1, 0, 'T', 2.1, [1,8], [1,10])

	:Notes:
	
		1. The spm8 and spm12 Matlab functions on which this code is based were
		developed by K.Friston and other members of the Wellcome Trust Centre for
		Neuroimaging. This function makes minor modifications to those procedures
		to take advantage of the simplicity of the 1D case.

		2. Results for the spm8 and spm12 versions can be obtained via the
		keyword "version". When expected ECs approach zero, the spm8 and spm12
		results will diverge slightly, due to a minor modification in spm12:
		In spm8: "EC = EC + eps".
		In spm12: "EC = np.array([max(ec,eps) for ec in EC])"

		3. Setting *c* and *k* in particular manners will yield important
		probabilities. Consider these three cases:
		
		(a)  rft(1, 0, STAT, Z, R)  --- field maximum
		(b)  rft(1, k, STAT, u, R)  --- cluster-based inference
		(c)  rft(c, k, STAT, u, R)  --- set-based inference
		
		(a) is the probability that Gaussian fields will produce 1 upcrossing
		with an extent of 0. Thus this pertains to the maximum of height of
		Gaussian fields, and can be used, for example, for critical threshold
		computations.
		
		(b) is the probability that Gaussian fields, when thresholded at *u*,
		will produce 1 upcrossing with an extent of *k*. This is used for
		cluster-level inference (i.e. p values for individual upcrossings).
		
		(c) is the probability that Gaussian fields, when thresholded at *u*,
		will produce *c* upcrossings with a minimum extent of *k*. This is
		used for set-level inference (i.e. p values for the entire result).
		
		.. warning:: Set-based inference (c) is more powerful than cluster-based inference (b), but unlike (b) it has no localizing information; it is a global p value pertaining to the entire excursion set en masse. It will thus always be lower than (b).

		4. If Q==None, then no Bonferroni check is made. If Q!=None, the RFT
		correction will be compared to Bonferroni correction, and the less
		severe correction will be returned. This will only have an effect
		for very rough fields, for example: when then second resel count
		approaches 0.5*Q.
	
	:References:
	
		1. Hasofer AM (1978) Upcrossings of random fields. Suppl Adv Appl
		   Prob 10:14-21.
		2. Friston KJ et al (1994) Assessing the significance of focal
		   activations using their spatial extent. Human Brain Mapping 1:
		   210-220.
		3. Worsley KJ et al (1996) A unified statistical approach for
		   determining significant signals in images of cerebral
		   activation. Human Brain Mapping 4:58-73.
	'''
	c        = _as_float(c)
	k        = _as_float(k)
	Z        = _as_float(Z)
	D        = float(len(R))  #dimensionality
	if R[1]==0:  #infinitely smooth field
		R = R[0], eps  #to make the results numerically stable
	R        = np.asarray(R, dtype=float)
	EC       = ec_density(STAT, Z, df)
	if version=='spm8':
		EC   = EC + eps
	elif version=='spm12':
		EC   = np.array([max(ec,eps) for ec in EC])
	else:
		raise( ValueError('rft1d error:  unknown version "%s" (version must be "spm8" or "spm12")'%str(version)) )
	if n==1:  #take a shortcut (Edit TCP 2014.08.11) -- about 9 times faster than the fast version below
		EM   = R*EC
		EN   = EC[0]*R[-1]
	else:
		### SLOW CODE -- but useful for D>1  (following spm8)
		# P = np.linalg.matrix_power( np.triu(linalg.toeplitz(EC*G)), n )
		# P = P[0,]
		### FASTER CODE (Edit TCP 2013.12.02) -- in 1D case this is about 25 times faster than using np.linalg.matrix_power
		# a,b   = EC*G
		# P     = a**n, n*b*a**(n-1)
		G    = sqrt(pi) / (gamma(0.5*np.arange(1,D+1)))
		a,b  = EC*G
		P    = a**n, n*b*a**(n-1)
		EM   = R/G*P
		EN   = P[0]*R[-1]
	### expected maxima and resels per cluster:
	Ec       = EM.sum()   #previously "Em"
	Ek       = EN/EM[-1]  #previously "En"
	if expectations_only:
		return Ec,Ek,EN
	### compute probabilities:  first P{n>k}
	D       -= 1
	if (k==0) or (D==0):
		p    = 1.0
	else:
		beta = (gamma(0.5*D+1)/Ek) **(2/D)
		p    = np.exp( -beta*(k**(2/D)) )
	#Poisson clumping heuristic (for multiple clusters)
	if p==0:
		P    = 0
	else:
		P    = 1 - poisson_cdf(c-1, (Ec + eps)*p)
	#Non-implemented cases:
	if version=='spm8':  #non-implemented flags are removed in spm12; rft1d validates all cases (see ./rft1d/examples/val*)
		if STAT in ['T','X2']:
			if (k>0) and (n>1):
				P,p   = None, None
		elif STAT=='F':
			if k>0:
				P,p   = None, None
	P        = _replaceWithBonferroniIfPossible(STAT, P, c, k, Z, df, Q, n)
	P        = _replaceWith0DpValueIfPossible(STAT, P, c, k, Z, df, Q, n)
	return P, p, Ec, Ek, EN



################################
# Crtical threshold computations
################################

def _approx_threshold(STAT, alpha, df, resels, n):
	# if two_tailed:
	# 	alpha   = 0.5*alpha
	a   = (alpha/sum(resels))**(1.0/n)
	if STAT=='Z':
		zstar = stats.norm.isf(a)
	elif STAT=='T':
		zstar = stats.t.isf(a, df[1])
	elif STAT=='X2':
		zstar = stats.chi2.isf(a, df[1])
	elif STAT=='F':
		zstar = stats.f.isf(a, df[0], df[1])
	elif STAT=='T2':
		p,m   = map(float,df)
		df_F  = p, m - p + 1
		fstar = stats.f.isf(a, df_F[0], df_F[1])
		zstar = fstar / ( (m-p+1)/(p*m) )
	else:
		raise(ValueError, 'Statistic must be one of: "Z", "T", "X2", "F", "T2"')
	return zstar

def isf(STAT, alpha, df, resels, n, Q=None, version='spm12'):
	'''
	Inverse survival function
	'''
	if isinstance(alpha, (int,float)):
		alpha       = [alpha]
	zstar = []
	for aaa in alpha:
		z0    = _approx_threshold(STAT, aaa, df, resels, n)
		fn    = lambda x : (rft(1, 0, STAT, x[0], df, resels, n, Q, False, version)[0] - aaa)**2
		zzz   = optimize.fmin(fn, z0, xtol=1e-9, disp=0)[0]
		zstar.append(zzz)
	return np.asarray(zstar)




################################
# Convienence classes
################################


def _as_float(x):
	if isinstance(x, (int,float)):
		x = float(x)
	elif isinstance(x, np.ndarray):
		x = np.asarray(x, dtype=float)
	return x


def _float_if_possible(x):
	if isinstance(x, np.ndarray):
		if x.size==1:
			return float( x.ravel()[0] )
		else:
			return x
	else:
		return x


class _Expected(object):
	def __init__(self, calc):
		self._calc  = calc
	def nodes_per_upcrossing(self, u):
		'''
		Number of nodes expected for each uprcrossing at threshold *u*.
		
		:Example:
	
			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.expected.nodes_per_upcrossing(2.7)
			
		.. warning:: This is a node count, so is equivalent to: (FWHM x **resels_per_upcrossing**)  + 1
		'''
		x = self._calc.FWHM * self.resels_per_upcrossing(u) + 1
		return _float_if_possible(x)
	def number_of_upcrossings(self, u):
		'''
		Number of upcrossings expected for threshold *u*.
		
		:Example:
	
			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.expected.number_of_upcrossings(2.7)
		'''
		x = self._calc._get_all(u, expectations_only=True)[:,0]
		return _float_if_possible(x)
	def number_of_suprathreshold_nodes(self, u):
		'''
		Number of nodes expected in the entire excursion set at threshold *u*.
		These nodes can come from multiple upcrossings. 

		:Example:
	
			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.expected.number_of_suprathreshold_nodes(2.8)
			
		.. warning:: This is a node count, so is equivalent to: (FWHM x **number_of_suprathreshold_resels**)  + **number_of_upcrossings**
		'''
		return self._calc.FWHM * self.number_of_suprathreshold_resels(u) + self.number_of_upcrossings(u)
	def number_of_suprathreshold_resels(self, u):
		'''
		Number of resels expected in the entire excursion set at threshold *u*.
		These resels can come from multiple upcrossings.
		One resel contains (1 x FWHM) nodes.
		Thus this is equivalent to:  (FWHM x number_of_suprathreshold_nodes)
		
		:Example:
	
			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.expected.number_of_suprathreshold_resels(2.9)
		'''
		x = self._calc._get_all(u, expectations_only=True)[:,2]
		return _float_if_possible(x)
	def resels_per_upcrossing(self, u):
		'''
		Number of nodes expected for each uprcrossing at threshold *u*.
		One resel contains (1 x FWHM) nodes.
		Thus this is equivalent to:  (FWHM x nodes_per_upcrossing)
		
		:Example:
	
			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.expected.resels_per_upcrossing(3.0)
		'''
		x = self._calc._get_all(u, expectations_only=True)[:,1]
		return _float_if_possible(x)




class _Probability(object):
	def __init__(self, calc):
		self._calc  = calc
	def cluster(self, k, u):
		'''
		Cluster-level inference.
		
		Probability that 1D Gaussian fields would produce an upcrossing of extent *k*
		when thresholded at *u*.
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **RFTCalculator.isf**
		
		:Parameters:

			*k* -- cluster extent (resels)
			
			*u* -- threshold

		:Returns:

			Cluster-specific probability value.

		:Examples:

			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.p.cluster(0.1, 3.0)
		'''
		return rft(1, k, self._calc.STAT, u, self._calc.df, self._calc.resels, self._calc.n, self._calc.Q, False, self._calc.version)[0]
		
	def set(self, c, k, u):
		'''
		Set-level inference.
		
		Probability that 1D Gaussian fields would produce at least *c* upcrossings
		with a minimum extent of *k* when thresholded at *u*.
		This probability pertains to the entire excursion set.
		
		.. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **RFTCalculator.isf**
		
		:Parameters:

			*c* -- number of upcrossings
			
			*k* -- minimum cluster extent (resels)
			
			*u* -- threshold

		:Returns:

			Set-specific probability value.

		:Examples:

			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.p.set(2, 0.1, 2.7)
		'''
		return rft(c, k, self._calc.STAT, u, self._calc.df, self._calc.resels, self._calc.n, self._calc.Q, False, self._calc.version)[0]
	
	def upcrossing(self, u):
		'''
		Survival function (equivalent to **RFTCalculator.sf**)
		
		Probability that 1D Gaussian fields would produce a 1D statistic field whose maximum exceeds *u*.

		:Parameters:

			*u* -- threshold (int, float, or sequence of int or float)

		:Returns:

			The probability of exceeding the specified heights.

		:Examples:

			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.sf(3.5)
		'''
		x = self._calc._get_all(u)[:,0]
		return _float_if_possible(x)





class RFTCalculator(object):
	'''
	A convenience class for high-level access to RFT probabilities.
	
	:Parameters:
	
		*STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")
		
		*df* --- degrees of freedom [df{interest} df{error}]
		
		*nodes* --- number of field nodes (int)  OR a binary field (boolean array)

		*FWHM* --- field smoothness (float)

		*n* --- number of test statistic fields in conjunction
		
		*withBonf* --- use a Bonferroni correction if less severe than the RFT correction
		
		*version* --- "spm8" or "spm12" (see below)
	
	:Returns:
	
		An instance of the RFTCalculator class.
	
	:Attributes:
	
		*expected* --- access to RFT expectations
		
		*p* --- access to RFT probabilities
		
	:Methods:
	
		*isf* --- inverse survival function
		
		*sf* --- survival function
	
	:Examples:
	
		>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
		>>> calc.expected.number_of_upcrossings(1.0) #yields 1.343
		>>> calc.expected.number_of_upcrossings(4.5) #yields 0.0223
	'''
	
	def __init__(self, STAT='Z', df=None, nodes=101, FWHM=10.0, n=1, withBonf=False, version='spm12'):
		self.FWHM     = None
		self.Q        = None
		self.STAT     = STAT
		self.df       = df
		self.mask     = None
		self.nNodes   = None
		self.n        = n
		self.resels   = None
		self.version  = 'spm12'
		self.withBonf = None
		self._parse_nodes_argument(nodes)
		self.set_fwhm(FWHM)
		self.set_bonf(withBonf)
		self.expected = _Expected(self)
		self.p        = _Probability(self)
		
	def __repr__(self):
		s    = ''
		s   += 'RFT1D RFTCalculator object:\n'
		s   += '   STAT     :  %s\n' %self.STAT
		s   += '   df       :  %s\n' %str(self.df)
		s   += '   nNodes   :  %d\n' %self.nNodes
		s   += '   FWHM     :  %.1f\n' %self.FWHM
		s   += '   withBonf :  %s\n' %self.withBonf
		return s
	
	def _get_all(self, u, expectations_only=False):
		if isinstance(u, (int,float)):
			u       = [u]
		return np.array([rft(1, 0, self.STAT, uu, self.df, self.resels, self.n, self.Q, expectations_only, self.version)   for uu in u])
	
	def _parse_nodes_argument(self, nodes):
		if isinstance(nodes, int):
			self.nNodes = nodes
		elif np.ma.is_mask(nodes):
			if nodes.ndim!=1:
				raise( ValueError('RFT1D Error:  the "nodes" argument must be a 1D boolean array. Received a %dD array'%arg.ndim)  )
			self.nNodes = nodes.size
			self.mask   = np.logical_not(nodes)
		else:
			raise( ValueError('RFT1D Error:  the "nodes" argument must be an integer or a 1D boolean array')  )
	
	def isf(self, alpha):
		'''
		Inverse survival function.
		(see also the survival function: **RFTCalculator.sf**)

		:Parameters:

			*alpha* -- upper tail probability (float;  0 < alpha < 1)

		:Returns:

			Quantile corresponding to upper-tail probability alpha.
			Equivalently: critical threshold at a Type I error rate of alpha.

		:Examples:

			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.isf(0.05)
		'''
		x = isf(self.STAT, alpha, self.df, self.resels, self.n, self.Q, self.version)
		return _float_if_possible(x)
	def set_bonf(self, wBonf):
		self.withBonf = bool(wBonf)
		self.Q        = float(self.nNodes) if self.withBonf else None
	def set_fwhm(self, w):
		self.FWHM   = float(w)
		if self.mask is None:
			self.resels = 1, (self.nNodes-1)/self.FWHM  #field length is (nNodes - 1)
		else:
			self.resels = geom.resel_counts(self.mask, fwhm=self.FWHM)
	def sf(self, u):
		'''
		Survival function.
		(Equivalent to **RFTCalculator.p.upcrossing**)
		
		Probability that 1D Gaussian fields with a smoothness *FWHM* would produce a 1D statistic field whose maximum exceeds *u*.

		:Parameters:

			*u* -- threshold (int, float, or sequence of int or float)

		:Returns:

			The probability of exceeding the specified heights.

		:Examples:

			>>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
			>>> calc.sf(3.5)
		'''
		return _float_if_possible(  self.p.upcrossing(u)  ) 



class RFTCalculatorResels(RFTCalculator):
	'''
	A convenience class for high-level access to RFT probabilities (based on resel counts).
	
	:Parameters:
	
		*STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")
		
		*df* --- degrees of freedom [df{interest} df{error}]
		
		*resels* --- resolution element counts

		*n* --- number of test statistic fields in conjunction
		
		*withBonf* --- use a Bonferroni correction if less severe than the RFT correction
		
		*nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)
		
		*version* --- "spm8" or "spm12" (see below)
	
	:Returns:
	
		An instance of the RFTCalculator class.
	
	:Attributes:
	
		*expected* --- access to RFT expectations
		
		*p* --- access to RFT probabilities
		
	:Methods:
	
		*isf* --- inverse survival function
		
		*sf* --- survival function
	
	:Examples:
	
		>>> calc = rft1d.prob.RFTCalculatorResels('T', (1,8), [1, 6.667])
		>>> calc.expected.number_of_upcrossings(1.0) #yields 1.343
		>>> calc.expected.number_of_upcrossings(4.5) #yields 0.0223
	'''
	
	def __init__(self, STAT='Z', df=None, resels=[1,10], n=1, withBonf=False, nNodes=None, version='spm12'):
		self.FWHM     = None
		self.Q        = None
		self.STAT     = STAT
		self.df       = df
		self.mask     = None
		self.nNodes   = nNodes
		self.n        = n
		self.resels   = tuple(resels)
		self.version  = 'spm12'
		self.withBonf = None
		self.set_bonf(withBonf)
		self.expected = _Expected(self)
		self.p        = _Probability(self)
		
	def __repr__(self):
		s    = ''
		s   += 'RFT1D RFTCalculatorResels object:\n'
		s   += '   STAT     :  %s\n' %self.STAT
		s   += '   df       :  %s\n' %str(self.df)
		s   += '   resels   :  (%d, %.3f)\n' %self.resels
		s   += '   FWHM     :  %.1f\n' %self.FWHM
		s   += '   withBonf :  %s\n' %self.withBonf
		return s
		
	def set_bonf(self, wBonf):
		self.withBonf = bool(wBonf)
		if self.withBonf and (self.nNodes is None):
			raise( ValueError('Must specify an integer value for "nNodes" when "withBonf" is True.') )
		self.Q        = float(self.nNodes) if self.withBonf else None


rftcalc  =  RFTCalculator()   #instantiated only for auto-doc generation
expected = _Expected(None)    #instantiated only for auto-doc generation
p        = _Probability(None) #instantiated only for auto-doc generation
