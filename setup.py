#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Jinja2==3.1.2",
    "matplotlib==3.7.0",
    "networkx==2.8.4",
    "numpy==1.23.5",
    "pandas==1.5.3",
    "scipy==1.10.0",
    "statsmodels==0.13.5",
    "tqdm==4.64.1",
    "animatplot==0.4.2",
]

test_requirements = []

setup(
    author="Uynaj GI",
    author_email="suquan12148@outlook.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A library project of Monte Carlo simulation algorithms for some statistical physics models (in particular, the Ising model and its variants).",
    entry_points={
        "console_scripts": [
            "mcmc_statphys=mcmc_statphys.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="mcmc_statphys",
    name="mcmc_statphys",
    packages=find_packages(include=["mcmc_statphys", "mcmc_statphys.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/uynajgi/mcmc_statphys",
    version="1.0.0",
    zip_safe=False,
)
