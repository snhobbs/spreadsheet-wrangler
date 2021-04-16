#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(name='spreadsheet-wrangler',
      version='0.1.1',
      description='Spreadsheet editing tools',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url='https://github.com/snhobbs/spreadsheet-wrangler',
      author='Simon Hobbs',
      author_email='simon.hobbs@electrooptical.net',
      license='MIT',
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires=[
          "click",
          "pandas",
          "numpy",
          #"json",
          #"pyxlsx"
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=["spreadsheet_wrangler.py"],
      entry_points={
        "console_scripts": [
            "spreadsheet_wrangler=spreadsheet_wrangler.__main__:main",
        ],
      },
      include_package_data=False,
      zip_safe=True)
