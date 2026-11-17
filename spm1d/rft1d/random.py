
'''
Functions and classes for generating 1D random fields.
The functions **randn1d** and **multirandn1d** are similar to the
**numpy.random.randn** and **np.random.multivariate_normal** functions.
If a large number of random fields are required (e.g. for RFT validations)
it may be more efficient to use the **Generator1D** and **GeneratorMulti1D** classes.
'''

# Copyright (C) 2026  Todd Pataky



from math import sqrt,log
import numpy as np
from scipy.ndimage import gaussian_filter1d


eps        = np.finfo(float).eps   #smallest float


class Generator1D(object):
	'''
	Generator of smooth Gaussian random fields.
	
	:Parameters:
	
		*nResponses* -- number of fields (int)
	
		*nodes* -- number of field nodes (int) OR a binary field (boolean array)
	
		*FWHM* -- field smoothness (float)
	
		*pad* -- pad prior to smoothing (bool)
	
	:Returns:
	
		A Generator1D object
		
	:Notes:
	
		1. Generator1D is faster than randn1d for iteratively generating many random samples.
	
	:Examples:
		
		>>> g = rft1d.random.Generator1D(8, 101, 15.0)
		>>> y = g.generate_sample()
		
	'''
	
	def __init__(self, nResponses=1, nodes=101, FWHM=10, pad=False):
		super(Generator1D, self).__init__()
		self.FWHM          = float(FWHM)
		self.SCALE         = None    #scale factor to return smoothed data to unit variance
		self.SD            = None    #standard deviation of the Gaussian kernel
		self.i0            = None    #first node, post-smoothed data
		self.i1            = None    #last node, post-smoothed data (i1-i0 = nodes)
		self.mask          = None
		self.nResponses    = int(nResponses)
		self.nNodes        = None
		self.pad           = bool(pad)
		self.q             = None    #number of nodes used for pre-smoothed data
		self._parse_nodes_argument(nodes)
		self.shape         = self.nResponses, self.nNodes
		self.set_fwhm(self.FWHM)
	
	def __repr__(self):
		s    = ''
		s   += 'RFT1D Generator1D:\n'
		s   += '   nResponses :  %d\n' %self.nResponses
		s   += '   nNodes     :  %d\n' %self.nNodes
		s   += '   FWHM       :  %.1f\n' %self.FWHM
		s   += '   pad        :  %s\n' %self.pad
		return s
	
	
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

	def _set_scale(self):
		'''
		Compute the scaling factor for restoring a smoothed curve to unit variance.

		Smoothing independent unit-variance noise with a kernel g yields noise
		with variance sum(g**2), so the scaling factor is 1 / sqrt( sum(g**2) ).

		Acknowledgement: this correction was first seen in "randomtalk.m" by
		Matthew Brett (Oct 1999), downloaded from
		http://www.fil.ion.ucl.ac.uk/~wpenny/mbi/index.html on 1 Aug 2014.
		The code below is implemented from the mathematics above.
		'''
		if np.isinf(self.FWHM):
			self.SCALE     = None
		else:
			t       = np.arange(  -0.5*(self.nNodes-1) , 0.5*(self.nNodes-1)+1  )
			g       = np.exp(-(t**2) / (2*self.SD**2 + eps))
			g      /= g.sum()
			self.SCALE = 1.0 / sqrt( (g**2).sum() )

	def _set_qi0i1(self, w):
		if np.isinf(w):
			self.q = self.i0 = self.i1 = None
		elif self.pad:
			n       = self.nNodes
			if w<3:
				q   = 2*n
			else:
				q   = 10*n
			if w>50:
				q  += n*(w-50)
			self.q  = int(q)
			self.i0 = self.q/2 - n/2
			self.i1 = self.i0 + n
		else:
			self.q  = self.nNodes
			self.i0 = 0
			self.i1 = self.nNodes
		self.i0     = int(self.i0)
		self.i1     = int(self.i1)

	def _smooth(self, y):
		return self.SCALE*gaussian_filter1d(y, self.SD, axis=1, mode='wrap')
	
	def generate_sample(self):
		if self.FWHM==0:
			y    = np.random.randn(*self.shape)
		elif np.isinf(self.FWHM):
			y    = np.random.randn(self.nResponses)
			y    = (   y*np.ones( tuple(self.shape) ).T   ).T
		else:
			y   = np.random.randn(self.nResponses, self.q)
			y   = self._smooth(y)
			y   = y[:,self.i0:self.i1]
		if self.mask is not None:
			y[:,self.mask] = np.nan
		return y

	def set_fwhm(self, fwhm):
		self.FWHM  = float(fwhm)
		self.SD    = self.FWHM / sqrt(8*log(2))
		self._set_scale()
		self._set_qi0i1(self.FWHM)



class GeneratorMulti1D(Generator1D):
	'''
	Generator of smooth multivariate Gaussian random fields.
	
	:Parameters:
	
		*nResponses* -- number of fields (int)
	
		*nodes* -- number of field nodes (int) OR a binary field (boolean array)
		
		*nComponents* -- number of vector components (int)
	
		*FWHM* -- field smoothness (float)
	
		*W* -- covariance matrix (*nComponents* x *nComponents* array)
		
		*pad* -- pad prior to smoothing (bool)
	
	:Returns:
	
		A GeneratorMulti1D object
		
	:Notes:
	
		1. GeneratorMulti1D is faster than multirandn1d for iteratively generating many random samples. 
	
	:Examples:
		
		>>> g = rft1d.random.GeneratorMulti1D(8, 101, 3, 15.0)
		>>> y = g.generate_sample()
		
	'''
	
	def __init__(self, nResponses=1, nodes=101, nComponents=2, FWHM=10, W=None, pad=False):
		super(GeneratorMulti1D, self).__init__(nResponses, nodes, FWHM, pad)
		self.nComponents   = int(nComponents)
		if W is None:
			self.W         = np.eye(self.nComponents)
		else:
			self.W         = np.asarray(W, dtype=float)
		self.shape         = self.nResponses, self.nNodes, self.nComponents
		self.mu            = np.array([0]*self.nComponents)
		
	def __repr__(self):
		s    = ''
		s   += 'RFT1D Generator1D:\n'
		s   += '   nResponses  :  %d\n' %self.nResponses
		s   += '   nNodes      :  %d\n' %self.nNodes
		s   += '   nComponents :  %d\n' %self.nComponents
		s   += '   FWHM        :  %.1f\n' %self.FWHM
		s   += '   W           :  (%dx%d array)\n' %self.W.shape
		s   += '   pad         :  %s\n' %self.pad
		return s
	
	
	def generate_sample(self):
		if self.FWHM==0:
			y   = np.random.multivariate_normal(self.mu, self.W, (self.nResponses,self.q))
		elif np.isinf(self.FWHM):
			y   = np.random.multivariate_normal(self.mu, self.W, (self.nResponses,))
			y   = np.dstack(  [   (yy*np.ones( (self.nResponses,self.nNodes) ).T).T   for yy in y.T] )
		else:
			y   = np.random.multivariate_normal(self.mu, self.W, (self.nResponses,self.q))
			y   = self._smooth(y)
			y   = y[:,self.i0:self.i1,:]
		if self.mask is not None:
			y[:,self.mask,:] = np.nan
		return y



def multirandn1d(nResponses, nodes, nComponents, FWHM=10.0, W=None, pad=False):
	'''
	Generate smooth Gaussian multivariate random fields.
	
	:Parameters:
	
		*nResponses* -- number of fields (int)
	
		*nodes* -- number of field nodes (int) OR a binary field (boolean array)
		
		*nComponents* -- number of vector components (int)
	
		*FWHM* -- field smoothness (float)
	
		*W* -- covariance matrix (*nComponents* x *nComponents* array)
		
		*pad* -- pad prior to smoothing (bool)
	
	:Returns:
	
		A 3D numpy array with shape:  (*nResponses*, *nodes*, *nComponents*)
		
	:Notes:
	
		1. The default *W* is the identity matrix.
		
		2. Padding is slow but necessary when 2 *FWHM* > *nodes*

	:Examples:
		
		>>> y = rft1d.random.multirandn1d(8, 101, 3, 15.0)
		>>> y = rft1d.random.multirandn1d(1000, 101, 5, 65.0, W=np.eye(5), pad=True)
	'''
	g     = GeneratorMulti1D(nResponses, nodes, nComponents, FWHM, W, pad)
	y     = g.generate_sample()
	return y


def randn1d(nResponses, nodes, FWHM=10.0, pad=False):
	'''
	Generate smooth Gaussian random fields.
	
	:Parameters:
	
		*nResponses* -- number of fields (int)
	
		*nodes* -- number of field nodes (int) OR a binary field (boolean array)
	
		*FWHM* -- field smoothness (float)
	
		*pad* -- pad prior to smoothing (bool)
		
	:Returns:
	
		A 2D numpy array with shape:  (*nResponses*, *nodes*)
		
	:Examples:
		
		>>> y = rft1d.random.randn1d(8, 101, 15.0)
		>>> y = rft1d.random.randn1d(1000, 101, 75.0, pad=True)
		
	.. warning:: Padding is slow but necessary when 2 *FWHM* > *nodes*
	'''
	g     = Generator1D(nResponses, nodes, FWHM, pad)
	y     = g.generate_sample()
	if nResponses==1:
		y = y.flatten()
	return y






