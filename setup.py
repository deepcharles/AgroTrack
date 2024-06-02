
# agrotrack/setup.py

from setuptools import setup, find_packages

setup(
    name='agrotrack',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'dask',
        'google-cloud',
        'ruptures',
        'scipy',
        'scikit-learn',
    ],
    author='Ehsan Jalilvand',
    author_email='ehsan.jalilvand@nasa.gov',
    description='A package to trace farmers’ irrigation decisions using satellite data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ejalilva/agrotrack',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
