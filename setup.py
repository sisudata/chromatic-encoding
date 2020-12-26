from pathlib import Path

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name="pycrank",
        version="0.0.0",
        url="https://github.com/sisudata/chromatic-encoding",
        author="vlad17",
        author_email="vlad@sisudata.com",
        packages=find_packages(include="pycrank*")
    )
