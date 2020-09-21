import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rann",
    version="1.0.1",
    author="bilkosem",
    author_email="bilkos92@gmail.com",
    description="Random Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bilkosem/random_neural_network",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)