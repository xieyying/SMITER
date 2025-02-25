#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Manuel Kösters",
    author_email="manuel.koesters@dcb.unibe.ch",
    maintainer="Yunying Xie",
    maintainer_email="xieyy@imb.pumc.edu.cn",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Library to create synthetic mzMLs file based on chemical formulas",
    entry_points={
        "console_scripts": [
            "SMITER_modified=smiter.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="SMITER_modified",
    name="SMITER_modified",
    packages=find_packages(include=["smiter", "smiter.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/yourusername/SMITER_modified",
    version="0.2.0",
    zip_safe=False,
)