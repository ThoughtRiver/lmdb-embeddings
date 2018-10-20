#!/usr/bin/env python
import os
from distutils.core import setup
from setuptools import find_packages

def get_readme():
    """ Get the package README as a string.

    :return str
    """
    root = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(root, 'README.md')

    with open(readme_path) as f:
        return f.read()

setup(
    name = 'lmdb_embeddings',
    version = '0.2.1',
    description = 'Fast querying of word embeddings using the LMDB "Lightning" Database.',
    license = 'GNU General Public License v3.0',
    long_description = get_readme(),
    long_description_content_type = 'text/markdown',
    author = 'Dom Hudson',
    author_email = 'dom.hudson@thoughtriver.com',
    url = 'https://www.thoughtriver.com',
    packages = find_packages(),
    install_requires = [
        'lmdb',
        'msgpack',
        'msgpack-numpy',
        'numpy',
        'pytest',
        'pytest-cov',
    ]
)
