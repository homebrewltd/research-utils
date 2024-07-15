# setup.py
from setuptools import setup, find_packages

setup(
    name='research-utils',
    version='0.2.0',  # Increment the version number
    description='A helper library for working with S3/Minio, Hugging Face models, and datasets',
    long_description='This library provides utilities for downloading and managing machine learning models and datasets from S3-compatible storage services, and loading them using the Hugging Face libraries.',
    author='Alan',
    author_email='alan@jan.ai',
    url='https://github.com/janhq/research-utils',
    packages=find_packages(),
    install_requires=[
        'boto3',
        'transformers',
        'datasets',  # Add the datasets library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    keywords='s3 minio huggingface transformers datasets machine-learning',
)