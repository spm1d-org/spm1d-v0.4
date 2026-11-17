# coding: utf-8

import os
import numpy as np
from .. import _base




class Dorn2012(_base.DatasetCCA, _base.Dataset1D):
	def _set_values(self):
		self.cite     = 'Dorn, T. W., Schache, A. G., & Pandy, M. G. (2012). Muscular strategy shift in human running: dependence of running speed on hip and ankle muscle performance. Journal of Experimental Biology, 215(11), 1944–1956. http://doi.org/10.1242/jeb.064527'
		self.www      = 'https://simtk.org/home/runningspeeds'
		self.note     = 'Results  ', 'Pataky TC, Vanrenterghem J, Robinson MA (2015). Zero- vs. one-dimensional, parametric vs. non-parametric, and confidence interval vs. hypothesis testing procedures in one-dimensional biomechanical trajectory analysis. Journal of Biomechanics 48 (7): 1277–1285. https://doi.org/10.1016/j.jbiomech.2015.02.051'
		self.datafile = os.path.join(_base.get_datafilepath(), 'Dorn2012.npz')
		### load and parse data:
		Z             = np.load(self.datafile)
		Y,FOOT,SPEED  = Z['Y'], Z['FOOT'], Z['SPEED']
		Z.close()
		### choose one foot:
		self.Y        = Y[FOOT==0]
		self.x        = SPEED[FOOT==0]
		self.z        = None
		self.df       = None
		self.p        = None



