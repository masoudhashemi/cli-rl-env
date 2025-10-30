from setuptools import setup, find_packages

setup(
    name="cli-rl-env",
    version="0.1.0",
    description="A Gymnasium environment for training LLMs to explore and edit code files",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pytest>=7.4.0",
        "pylint>=3.0.0",
        "flake8>=6.0.0",
        "pytest-timeout>=2.2.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8.1",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

