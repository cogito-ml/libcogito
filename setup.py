
from setuptools import setup, find_packages

setup(
    name="libcogito",
    version="0.1.0",
    description="Python bindings for the Cogito ML library",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
)
