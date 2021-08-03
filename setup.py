import os
from glob import glob
from setuptools import setup

exec(open('counterstone/version.py').read())

setup(name='counterstone',
      version=__version__,
      description='Counterfactual generation with STONED SELFIES',
      author='Aditi Seshadri, Geemi Wellawatte, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='github.com/ur-whitelab/counterstone',
      license='MIT',
      packages=['counterstone'],
      install_requires=[
          'selfies',
          'numpy',
          'rdkit-pypi',
          'matplotlib'],
      test_suite='tests',
      zip_safe=True
      )
