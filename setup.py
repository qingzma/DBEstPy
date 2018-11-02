# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dbest',
    version='2.0',
    description='Model-based Approximate Query Processing (AQP) engine.',
    url='https://github.com/qingzma/DBEst',
    author='Qingzhi Ma',
    author_email='Q.Ma.2@warwick.ac.uk',
    long_description=readme,
    license=license,
    packages=['dbest'],#find_packages(exclude=('tests', 'docs','results'))
    zip_safe=False
)

