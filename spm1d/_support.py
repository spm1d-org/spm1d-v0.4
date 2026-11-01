
'''
System information for bug reports.

A bug report is actionable only when the versions that produced it are known.
"spm1d.get_config" collects them in one call in the layout the report form
expects.

The machine's name, the account name and the file paths that contain them are
not collected because a bug report is public.
'''

# Copyright (C) 2026  Todd Pataky


import os
import platform
import sys


#  Packages reported for every configuration in this order.
_PACKAGES = ('spm1d', 'numpy', 'scipy', 'matplotlib')


def _version(name):
	'''
	Version of a package, preferring the copy that is actually running.

	An imported module answers for itself through "__version__". That is the
	version of the code that produced the result being reported. Installation
	metadata is consulted only for packages that are not imported so that
	matplotlib's version is reported without importing matplotlib. Metadata
	describes what is installed. That differs from the running copy when a
	source tree is run alongside an older release.
	'''
	mod = sys.modules.get(name)
	v   = None if (mod is None) else getattr(mod, '__version__', None)
	if v is not None:
		return str(v)
	try:
		from importlib.metadata import PackageNotFoundError, version
	except ImportError:  # Python 3.7 has no importlib.metadata
		return 'unknown'
	try:
		return version(name)
	except PackageNotFoundError:
		return 'not installed'


def _install_kind():
	'''
	Whether this copy was installed or is being run from a source tree.

	Decided from the location of this file. The location itself is not reported
	because it usually contains the account name.
	'''
	parts = os.path.abspath( __file__ ).split( os.sep )
	inst  = ('site-packages' in parts) or ('dist-packages' in parts)
	return 'installed' if inst else 'source'


class Config(object):
	'''
	A snapshot of the versions and the platform a result was produced under.

	Printing it gives the block that the bug report form asks for. The same
	values are available through "asdict".
	'''

	def __init__(self, packages=None):
		names         = list( _PACKAGES if (packages is None) else packages )
		#: Reported package names in order.
		self.names    = names
		#: Version string per package name (``"not installed"`` when absent).
		self.packages = dict( [(s, _version(s)) for s in names] )
		#: Python version and implementation.
		self.python   = '%d.%d.%d (%s)' % (sys.version_info[:3]
										   + (platform.python_implementation(),))
		#: Operating system and machine.
		self.platform = platform.platform()
		#: ``"installed"`` (from site-packages) or ``"source"`` (a checkout).
		self.install  = _install_kind()

	def __repr__(self):
		return self.asstr()

	def _rows(self):
		rows = [(s, self.packages[s]) for s in self.names]
		rows.append( ('python',   self.python) )
		rows.append( ('platform', self.platform) )
		rows.append( ('install',  self.install) )
		return rows

	def asdict(self):
		'''
		Same values as a dictionary.

		:Returns:

		- A dictionary with one entry per reported package plus ``python``,
		  ``platform`` and ``install``

		:Example:

			>>> import spm1d
			>>> config = spm1d.get_config()
			>>> print( sorted( config.asdict().keys() )[:4] )
			['install', 'matplotlib', 'numpy', 'platform']
		'''
		d = dict( self.packages )
		d['python']   = self.python
		d['platform'] = self.platform
		d['install']  = self.install
		return d

	def asstr(self):
		'''
		Printed form with one "name value" pair per line and aligned columns.

		:Returns:

		- The block as a string. ``print( config )`` prints the same block

		:Example:

			>>> import spm1d
			>>> config = spm1d.get_config()
			>>> print( config.asstr().splitlines()[0].split()[0] )
			spm1d
		'''
		rows = self._rows()
		n    = max( [len(k) for k, v in rows] )
		return '\n'.join( ['%-*s  %s' % (n, k, v) for k, v in rows] )


def get_config(packages=None):
	'''
	Versions and platform for a bug report.

	Run this and paste the result into the "Environment" field of the bug
	report form.

	:Parameters:

	- *packages* --- sequence of additional package names to report (default
	  None)

	:Returns:

	- A **Config** object. Printing it gives the block that the form expects.
	  Its ``asdict`` method gives the same values as a dictionary

	:Example:

		>>> import spm1d
		>>> spm1d.get_config()
		spm1d       0.4.55
		numpy       2.1.0
		scipy       1.14.0
		matplotlib  3.9.0
		python      3.12.4 (CPython)
		platform    macOS-15.1-arm64-arm-64bit
		install     installed

	:Note:

	- The machine name, the account name and file paths are not reported
	  because a bug report is public.
	'''
	names = list( _PACKAGES )
	if packages is not None:
		names += [s for s in packages if s not in names]
	return Config( names )
