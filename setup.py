"""
Setup script for Neural Market Microstructure Predictor.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-market-predictor",
    version="1.0.0",
    author="Utkarsh Upadhyay",
    author_email="utkarsh.upadhyay9@example.com",
    description="A neural network-based system for market microstructure prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Utkarsh-upadhyay9/neural-market-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "train-model=scripts.train_model:main",
            "run-predictions=scripts.run_predictions:main",
        ],
    },
)
