from setuptools import setup, find_packages

setup(
    name='gi',
    version='0.0.1',
    description='Paint the frame',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
