import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="cubo",
    version="2024.8.0",
    url="https://github.com/davemlz/cubo",
    license="MIT",
    author="David Montero Loaiza",
    author_email="dml.mont@gmail.com",
    description="On-Demand Earth System Data Cubes in Python",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    package_data={"cubo": ["data/*.json"]},
    install_requires=[
        "dask>=2023.7.0",
        "numpy",
        "pandas>=2.0.3",        
        "planetary_computer>=1.0.0",
        "pystac_client>=0.7.2",
        "scipy>=1.12.0",
        "stackstac>=0.5.1",
        "xarray>=2023.6.0",
    ],
    extras_require={
        "ee": ["xee>=0.0.6"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
