# setup.py
from setuptools import setup, find_packages

setup(
    name='research-utils',
    version='0.1.0',
    description='A helper library for working with S3 and Hugging Face models',
    author='Alan',
    author_email='alan@jan.ai',
    url='https://github.com/janhq/research-utils',
    packages=find_packages(),
    install_requires=[
        'boto3',
        'transformers',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

