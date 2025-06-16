from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="npd-simulation",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="N-Person Prisoner's Dilemma Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/npd-simulation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Core functionality has no dependencies
    ],
    extras_require={
        "viz": ["numpy>=1.21.0", "matplotlib>=3.4.0", "seaborn>=0.11.0", "pandas>=1.3.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
    },
    entry_points={
        "console_scripts": [
            "npd-sim=main:main",
        ],
    },
)