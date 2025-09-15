#!/usr/bin/env python3
"""
PyQuotex AI Trading Bot Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    with open("pyquotex/__init__.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "2.0.0"

setup(
    name="pyquotex-ai",
    version=get_version(),
    author="Team Jund Al Nabi",
    author_email="alet8319@gmail.com",
    description="AI-Powered Trading Bot with Neural Network for Quotex Platform",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyquotex",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pyquotex/issues",
        "Source": "https://github.com/yourusername/pyquotex",
        "Documentation": "https://github.com/yourusername/pyquotex/blob/main/README.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyquotex=app:main",
            "pyquotex-ai=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pyquotex": [
            "*.py",
            "http/*.py",
            "utils/*.py",
            "ws/*.py",
            "ws/channels/*.py",
            "ws/objects/*.py",
        ],
    },
    keywords=[
        "trading",
        "bot",
        "ai",
        "neural-network",
        "machine-learning",
        "quotex",
        "forex",
        "binary-options",
        "automation",
        "finance",
    ],
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    maintainer="Team Jund Al Nabi",
    maintainer_email="alet8319@gmail.com",
    download_url="https://github.com/yourusername/pyquotex/archive/v2.0.0.tar.gz",
    provides=["pyquotex"],
    requires_python=">=3.8",
    long_description_content_type="text/markdown",
    data_files=[
        ("", ["LICENSE", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SETUP.md"]),
    ],
)
