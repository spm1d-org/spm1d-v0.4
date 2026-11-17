# coding: utf-8

import os
import numpy as np
from .. import _base




class SpeedGRFcategorical(_base.DatasetANOVA1, _base.Dataset1D):
	def __init__(self, subj=0):
		self.subj     = int(subj)
		super(SpeedGRFcategorical, self).__init__()
		
	
	def _set_values(self):
		self.cite     = 'Pataky, T. C., Caravaggi, P., Savage, R., Parker, D., Goulermas, J., Sellers, W., & Crompton, R. (2008). New insights into the plantar pressure correlates of walking speed using pedobarographic statistical parametric mapping (pSPM). Journal of Biomechanics, 41(9), 1987–1994.'
		fnameY        = os.path.join(_base.get_datafilepath(), 'ex_grf_subj%03d.npy'%self.subj)
		fnameX        = os.path.join(_base.get_datafilepath(), 'ex_grf_speeds_cond.npy')
		self.datafile = fnameY
		self.Y        = np.load(fnameY)
		self.A        = np.load(fnameX)[:,self.subj]
		self.z        = None
		self.df       = None
		self.p        = None



class Weather(_base.DatasetANOVA1, _base.Dataset1D):
	def _set_values(self):
		self.cite     = 'Ramsay JO, Silverman BW (2005). Functional Data Analysis (Second Edition), Springer, New York.'
		self.www      = 'https://cran.r-project.org/package=fda'
		fname         = os.path.join(_base.get_datafilepath(), 'weather.npz')
		Z             = np.load(fname)
		y0,y1,y2,y3   = Z['y0'], Z['y1'], Z['y2'], Z['y3']
		Z.close()
		self.datafile = fname
		self.Y        = np.vstack([y0,y1,y2,y3])
		self.A        = np.array( [0]*y0.shape[0] + [1]*y1.shape[0] + [2]*y2.shape[0] + [3]*y3.shape[0] )
		self.z        = None
		self.df       = None
		self.p        = None


