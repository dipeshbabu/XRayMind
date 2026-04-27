from setuptools import find_packages, setup

setup(
    name="xraymind",
    version="0.3.0",
    description="Explainable chest X-ray inference, reporting, and reliability evaluation utilities for research demos.",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12",
        "torchvision>=0.13",
        "torchxrayvision>=0.0.38",
        "captum>=0.5.0",
        "gradio>=4.0",
        "matplotlib>=3.2",
        "scikit-image>=0.16",
        "scikit-learn>=1.0",
        "numpy>=1",
        "pandas>=1",
        "pillow>=8.0",
        "requests>=1",
        "tqdm>=4",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "xraymind=xraymind.cli:main",
        ]
    },
)
