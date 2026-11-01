

'''
spm1D: A Python package for 1D Statistical Parametric Mapping

Useful references include:

[1] Statistical Parametric Mapping: The Analysis of Functional Brain Images
Karl Friston, John Ashburner, Stefan Kiebel, Thomas Nichols (Editors)
Academic Press, 2006

[2] Wellcome Trust Centre for Neuroimaging SPM homepage:
www.fil.ion.ucl.ac.uk/spm/

[3] Pataky TC (2016). RFT1D: Smooth one-dimensional random field upcrossing
probabilities in Python, Journal of Statistical Software 71(7), 1–22.


Copyright (C) 2026  Todd Pataky
'''


__version__ = '0.4.56'  # 2026-11-01


__all__ = ['data', 'io', 'plot', 'rft1d', 'stats', 'util', 'get_config']

from . import data
from . import geom
from . import io
from . import plot
from . import rft1d
from . import stats
from . import util
from . _support import get_config

