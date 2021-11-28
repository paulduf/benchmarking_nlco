import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlcco",
    version="0.0.3",
    author="Paul Dufossé",
    author_email="paul.dufosse@inria.fr",
    description="Benchmark Problems for Non Linear Constrained Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    requires=["numpy", "matplotlib", "codetiming"],
    python_requires='>=3.6',
)
