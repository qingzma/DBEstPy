# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dbest',
    version='2.1',
    description='Model-based Approximate Query Processing (AQP) engine.',
    classifiers=[
        'Development Status :: 2.0',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Approximate Query Processing :: AQP :: Data Warehouse',
      ],
    keywords='Approximate Query Processing AQP',
    url='https://github.com/qingzma/DBEst',
    author='Qingzhi Ma',
    author_email='Q.Ma.2@warwick.ac.uk',
    long_description=readme,
    license=license,
    packages=['dbest'],#find_packages(exclude=('tests', 'docs','results'))
    zip_safe=False,
    install_requires=[
          'xgboost','numpy','dill'
      ],
    test_suite='nose.collector',
    tests_require=['nose'],
)

