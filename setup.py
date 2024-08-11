from setuptools import setup

setup(
    packages=["ai4rgym","policies","unit_tests","evaluation"],
    name="ai4rgym",
    version="0.0.1",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.23.0,<2.0",
        "matplotlib>=3.0.0",
        "scipy>=1.6.0",
    ],
)

# Note:
# This is a useful example that describes each of
# the possible options for the "setup" function:
# https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py