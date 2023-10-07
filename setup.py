from setuptools import setup, find_packages

setup(
    name='scanorama',
    version='1.7.4',
    description='Panoramic stitching of heterogeneous single cell transcriptomic data',
    url='https://github.com/brianhie/scanorama',
    packages=find_packages(exclude=['bin', 'conf', 'data', 'target']),
    install_requires=[
        'annoy>=1.11.5',
        'fbpca>=1.0',
        'geosketch>=1.0',
        'intervaltree>=3.1.0',
        'matplotlib>=2.0.2',
        'numpy>=1.12.0',
        'scipy>=1.0.0',
        'scikit-learn>=0.20rc1',
    ],
    author='Brian Hie',
    author_email='brianhie@stanford.edu',
    license='MIT'
)
