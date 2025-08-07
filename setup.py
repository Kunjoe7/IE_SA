#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='icdm-sa',
    version='0.1.0',
    author='ICDM-SA Team',
    author_email='',
    description='Interactive and Explainable Survival Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Kingmaoqin/ICDM-SA',
    packages=find_packages(),
    package_data={
        'icdm_sa': [
            'datasets/flchain/*.csv',
            'datasets/gabs/*.csv',
            'datasets/metabric/*.csv',
            'datasets/nwtco/*.csv',
            'datasets/tcga_task3/*.csv',
        ],
    },
    include_package_data=True,
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'scikit-survival>=0.18.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.62.0',
        'easydict>=1.9',
        'lifelines>=0.27.0',
        'scipy>=1.7.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='survival analysis machine learning deep learning explainable ai',
    project_urls={
        'Bug Reports': 'https://github.com/Kingmaoqin/ICDM-SA/issues',
        'Source': 'https://github.com/Kingmaoqin/ICDM-SA',
    },
)