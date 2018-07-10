"""
 Setup tools configuration for installing this package
"""
from setuptools import setup, find_packages

setup(
    name='prob_mbrl',
    version='0.1',
    description='A library for probabilistic model based rl algorithms',
    url='http://github.com/juancamilog/prob_mbrl',
    author='Juan Camilo Gamboa Higuera',
    author_email='juancamilog@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'scipy', 'matplotlib', 'dill'],
    )
