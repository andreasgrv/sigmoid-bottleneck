import os
from glob import glob
from itertools import chain
from setuptools import setup


def collect(*patterns):
    return set(chain(*(filter(os.path.isfile, glob(p))
                     for p in patterns)))


setup(
      name='spmlbl',
      packages=['spmlbl'],
      version='0.0.1',
      author='CHANGE_ME',
      author_email='CHANGE_ME',
      description='Approaches for multi-label classification',
      license='BSD',
      keywords=['classification', 'multilabel'],
      classifiers=[],
      scripts=collect(
        'bin/process/*',
        'bin/visualise/*'
      ),
      install_requires=[],
      tests_require=['pytest', 'pytest-sugar', 'pytest-xdist']
      )
