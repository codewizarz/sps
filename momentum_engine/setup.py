from setuptools import setup, find_packages

setup(
    name="momentum_engine",
    version="0.1.0",
    description="Momentum Engine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "yfinance",
        "ta",
        "requests",
    ],
)
