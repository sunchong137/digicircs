from setuptools import setup, find_packages
import os
import sys

setup(
    name='digicircs',
    install_requires=[],
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        '': [os.path.join('.')]
    }
)
