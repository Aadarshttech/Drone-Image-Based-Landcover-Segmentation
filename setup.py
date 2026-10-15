from setuptools import setup, find_packages

setup(
    name="drone-landcover-segmentation",
    version="1.0.0",
    description="Deep learning pipeline for high-resolution drone imagery landcover segmentation",
    author="Aadarsh Pandita, Aayush Gautam",
    author_email="aadarshapandit@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
