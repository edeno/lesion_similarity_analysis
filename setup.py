#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'scipy', 'xarray', 'pandas',
                    'scikit-image', 'dask', 'napari']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='lesion_similarity_analysis',
    version='0.1.0.dev0',
    license='MIT',
    description=('David Kastner, Greer Williams'),
    author='',
    author_email='',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
