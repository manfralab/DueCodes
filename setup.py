from setuptools import setup, find_packages

setup(
    name='sqpurdue',
    version='0.1',
    description='wrappers for helping to run an experiment with QCoDeS',
    url='https://github.com/manfralab/mmsweeps',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Licence :: MIT Licence',
        'Topic :: Scientific/Engineering'
    ],
    license='MIT',
    packages=find_packages(),
    python_requires='>=3'
)
