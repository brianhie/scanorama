from setuptools import setup, find_packages

setup(
    name='scanorama',
    version='0.2',
    description='Panoramic stitching of heterogeneous single cell transcriptomic data',
    url='https://github.com/brianhie/scanorama',
    packages=find_packages(exclude=['bin', 'conf', 'data']),
    install_requires=[
        'annoy>=1.11.5',
        'fbpca>=1.0',
        'intervaltree>=2.1.0',
        'matplotlib>=2.0.2',
        'networkx>=2.1',
        'numpy>=1.12.0',
        'scipy>=1.0.0',
        'scikit-learn>=0.19.0',
        'statsmodels>=0.8.0rc1',
        'tables>=3.3.0'
    ],
    author='Brian Hie',
    author_email='brianhie@mit.edu',
    license='MIT'
)
