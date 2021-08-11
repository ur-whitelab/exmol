import os
from glob import glob
from setuptools import setup

exec(open('exmol/version.py').read())

setup(name='exmol',
      version=__version__,
      description='Counterfactual generation with STONED SELFIES',
      author='Aditi Seshadri, Geemi Wellawatte, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='github.com/ur-whitelab/exmol',
      license='MIT',
      packages=['exmol', 'exmol.stoned'],
      install_requires=[
          'selfies',
          'numpy',
          'rdkit-pypi',
          'matplotlib',
          'scikit-learn'],
      test_suite='tests',
      zip_safe=True
      )
