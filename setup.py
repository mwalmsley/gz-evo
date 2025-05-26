from setuptools import setup, find_packages

setup(
    name="gz_evo",
    version="0.1.0",
    description="Galaxy Zoo Evolution baseline training and evaluation package",
    author="Mike Walmsley",
    author_email="m.walmsley@utoronto.ca",
    packages=find_packages(),
    install_requires=[
        "omegaconf",
        "wandb",
        "pytorch-lightning",
        "torch",
        "zoobot",
        "datasets",
        "pandas",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)