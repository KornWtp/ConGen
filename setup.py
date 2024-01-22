from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="congen-sbert",
    version="1.0.0",
    author=" ",
    author_email=" ",
    description="Sentence representation with SBERT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "transformers==4.9.0",
        "sentence-transformers==2.0.0",
    ],
)
