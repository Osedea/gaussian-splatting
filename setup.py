from setuptools import setup, find_packages

setup(
    name='gaussian_splatting',
    version='0.1.0',
    packages=find_packages(),  # Automatically discover and include all packages
    scripts=[],
    install_requires=[
        "pyyaml",
        "black",
        "isort",
        "importchecker",
        "matplotlib",
        "transformers",
        "chardet"
    ]
)
