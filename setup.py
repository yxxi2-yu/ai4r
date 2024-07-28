from setuptools import setup

setup(
    packages=["ai4rgym","policies"],
    name="ai4rgym",
    version="0.0.1",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.23.0",
        "matplotlib>=3.0.0",
        "scipy>=1.6.0",
    ],
)