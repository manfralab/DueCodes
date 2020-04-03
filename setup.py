from setuptools import setup, find_packages

setup(
    name="duecodes",
    version="0.1",
    description="wrappers for helping to run an experiment with QCoDeS",
    url="https://dev.azure.com/manfra-lab/DueCodes",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Licence :: MIT Licence",
        "Topic :: Scientific/Engineering",
    ],
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'matplotlib>=3.0.0',
        'pandas>=0.22',
        'xarray',
    ],
)
