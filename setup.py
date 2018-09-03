#!/usr/bin/env python
import os
from distutils.core import setup
from setuptools import find_packages


setup(
    name = 'lmdb_embeddings',
    version = '0.0.1',
    description = 'Fast querying of word embeddings using the LMDB "Lightning" Database.',
    author = 'Dom Hudson',
    author_email = 'dom.hudson@thoughtriver.com',
    url = 'https://www.thoughtriver.com',
    packages = find_packages(),
    install_requires = [
        'lmdb',
        'numpy',
        'pytest',
        'pytest-cov',
    ]
)