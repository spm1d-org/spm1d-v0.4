
from setuptools import setup



long_description = '''
**spm1d** is a Python package for <b>one-dimensional Statistical Parametric Mapping</b>.
spm1d uses <b>Random Field Theory</b> expectations regarding the behavior of smooth,
one-dimensional Gaussian fields to make statistical inferences regarding a set of
one-dimensional measurements.
'''

setup(
    name             = 'spm1d',
    version          = '0.4.59',
    description      = 'One-Dimensional Statistical Parametric Mapping',
    author           = 'Todd Pataky',
    author_email     = 'spm1d.mail@gmail.com',
    url              = 'https://github.com/spm1d-org/spm1d-v0.4',
    license          = 'GPL-3.0-or-later',
    download_url     = 'https://github.com/spm1d-org/spm1d-v0.4/archive/master.zip',
    packages         = ['spm1d'],
    package_data     = {'spm1d' : ['examples/*.*', 'data/*.*'] },
    include_package_data = True,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    python_requires  = '>=3.10',
    keywords         = ['statistics', 'time series analysis'],
    classifiers      = [
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    install_requires = ["numpy", "scipy", "matplotlib"]
)