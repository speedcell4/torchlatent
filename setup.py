from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name='torchlatent',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/speedcell4/torchlatent',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='A latent structure inference library',
    long_description=long_description,
    python_requires='>=3.7',
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
    }
)
