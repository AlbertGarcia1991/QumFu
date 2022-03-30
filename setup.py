import re
import sys
import warnings

import versioneer
from __init__ import __version__
from setuptools import find_packages, setup

# Get current package version
FULLVERSION = versioneer.get_version()
ISRELEASED = re.search(r"(dev|\+)", FULLVERSION) is None
_V_MATCH = re.match(r"(\d+)\.(\d+)\.(\d+)", FULLVERSION)
if _V_MATCH is None:
    raise RuntimeError(f"Cannot parse version {FULLVERSION}")
MAJOR, MINOR, MICRO = _V_MATCH.groups()
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

# Check Python version is at least 3.7
if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

# Package has only been tested until Python 3.9
if sys.version_info >= (3, 10):
    fmt = "QumFu {} may not yet support Python {}.{}."
    warnings.warn(fmt.format(VERSION, *sys.version_info[:2]), RuntimeWarning)
    del fmt

setup(
    name="QumFu",
    version=__version__,
    license="GPLv3",
    author="Albert Garcia",
    author_email="gplaza91@gmail.com",
    packages=find_packages("."),
    package_dir={"": "src"},
    url="https://github.com/AlbertGarcia1991/QumFu",
    keywords=[
        "python",
        "optimization",
        "function optimization",
        "scikit-learn",
        "rapids",
        "tensorflow",
    ],
    install_requires=[
        "numpy~=1.20.3",
        "scipy~=1.7.1",
        "setuptools~=58.0.4",
        "versioneer~=0.22",
    ],
)
