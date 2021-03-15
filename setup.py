from setuptools import setup, find_packages

name = 'torchlatent'

setup(
    name=name,
    version='0.1.0',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchlatent',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='High Performance Structured Prediction in PyTorch',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'torchrua',
        'torchglyph',
    ],
    extras_require={
        'dev': [
            'einops',
            'pytest',
            'hypothesis',
            'pytorch-crf',
        ],
        'benchmark': [
            'aku',
            'tqdm',
            'pytorch-crf',
        ]
    }
)
