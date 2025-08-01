"""Setup configuration for ONIKS NeuralNet Framework."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="oniks-neuralnet",
    version="0.1.0",
    author="Horlov Danylo",
    author_email="your.email@example.com",  # Replace with actual email
    description="A robust, deterministic, multi-agent framework for complex task automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ONIKS_NeuralNet",  # Replace with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    },
    entry_points={
        "console_scripts": [
            "oniks-demo=oniks.cli:run_demo",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, multi-agent, llm, automation, planning, execution, framework",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ONIKS_NeuralNet/issues",
        "Source": "https://github.com/your-username/ONIKS_NeuralNet",
        "Documentation": "https://github.com/your-username/ONIKS_NeuralNet#readme",
    },
)