from setuptools import find_packages
from setuptools import setup

name = 'torchlatent'

setup(
    name=name,
    version='0.4.2',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchlatent',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='High Performance Structured Prediction in PyTorch',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'torchrua>=0.4.0',
    ],
    extras_require={
        'dev': [
            'einops',
            'pytest',
            'hypothesis',
            'pytorch-crf',
        ],
    }
)
