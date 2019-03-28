import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="features-pkg-ayeaton",
    version="0.0.1",
    author="Anna Yeaton",
    author_email="ahyeaton@gmail.com",
    description="A package to hold pytorch code for the histo path project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayeaton/features",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
