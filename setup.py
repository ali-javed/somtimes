#!/usr/bin/env python
import setuptools

__version__ = "1.0.0"


CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.2",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setuptools.setup(
    name="somtimes",
    version=__version__,
    description="Python implementation of SOMTimeS: Self Organizing Map for Time Series",
    long_description = "",
    author="A. Javed",
    author_email="alijaved@live.com",
    packages=setuptools.find_packages(),
    zip_safe=True,
    license="",
    download_url = "https://github.com/ali-javed/somtimes/archive/1.0.8.tar.gz",
    url="https://github.com/ali-javed/somtimes",
    install_requires=['numpy','matplotlib']
)
