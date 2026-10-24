
from setuptools import setup



long_description = '''
**spm1d** is a Python package for <b>one-dimensional Statistical Parametric Mapping<b>.
spm1d uses <b>Random Field Theory</b> expectations regarding the behavior of smooth,
one-dimensional Gaussian fields to make statistical inferences regarding a set of
one-dimensional measurements.
'''

setup(
    name             = 'spm1d',
    version          = '0.4.55',
    description      = 'One-Dimensional Statistical Parametric Mapping',
    author           = 'Todd Pataky',
    author_email     = 'spm1d.mail@gmail.com',
    url              = 'https://github.com/0todd0000/spm1d',
    download_url     = 'https://github.com/0todd0000/spm1d/archive/master.zip',
    packages         = ['spm1d'],
    package_data     = {'spm1d' : ['examples/*.*', 'data/*.*'] },
    include_package_data = True,
    long_description = long_description,
    keywords         = ['statistics', 'time series analysis'],
    classifiers      = [],
    install_requires = ["numpy", "scipy", "matplotlib"]
)