# coding: utf-8

import os
import numpy as np
from .. import _base




class Neptune1999kneekin(_base.DatasetHotellingsPaired, _base.Dataset1D):
	def _set_values(self):
		self.cite     = 'Neptune, R. R., Wright, I. C., & van den Bogert, A. J. (1999). Muscle coordination and function during cutting movements. Medicine & Science in Sports & Exercise, 31(2), 294–302.'
		self.www      = 'http://isbweb.org/data/rrn/index.html'
		self.note     = 'Results  ', 'Pataky TC, Vanrenterghem J, Robinson MA (2015). Zero- vs. one-dimensional, parametric vs. non-parametric, and confidence interval vs. hypothesis testing procedures in one-dimensional biomechanical trajectory analysis. Journal of Biomechanics 48 (7): 1277–1285. https://doi.org/10.1016/j.jbiomech.2015.02.051'
		self.datafile = os.path.join(_base.get_datafilepath(), 'Neptune1999kneekin.npz')
		Z             = np.load(self.datafile)
		self.YA       = Z['YA']
		self.YB       = Z['YB']
		Z.close()
		self.z        = None
		self.df       = None
		self.p        = None


class Pataky2014cop(_base.DatasetHotellingsPaired, _base.Dataset1D):
	def _set_values(self):
		self.cite     = 'Pataky, T. C., Robinson, M. A., Vanrenterghem, J., Savage, R., Bates, K. T., & Crompton, R. H. (2014). Vector field statistics for objective center-of-pressure trajectory analysis during gait, with evidence of scalar sensitivity to small coordinate system rotations. Gait and Posture, 1–4. http://doi.org/10.1016/j.gaitpost.2014.01.023'
		self.datafile = os.path.join(_base.get_datafilepath(), 'Pataky2014cop.npz')
		Z             = np.load(self.datafile)
		self.YA       = Z['YA']
		self.YB       = Z['YB']
		Z.close()
		self.z        = None
		self.df       = None
		self.p        = None


