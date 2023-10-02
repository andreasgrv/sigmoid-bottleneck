import os
from glob import glob
from itertools import chain
from setuptools import setup


def collect(*patterns):
    return set(chain(*(filter(os.path.isfile, glob(p))
                     for p in patterns)))


setup(
      name='asl',
      packages=['asl'],
      version='0.0.1',
      # scripts=collect(
      #   'bin/*',
      #   'bin/process/*',
      #   'bin/visualise/*'
      # ),
      # install_requires=[],
      # tests_require=['pytest', 'pytest-sugar', 'pytest-xdist']
      )
