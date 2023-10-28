#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'pandas',
    'pyexcel_ods3',
    'openpyxl'
    ]


test_requirements = [ ]

setup(
    author="Simon Hobbs",
    author_email='simon.hobbs@electrooptical.net',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Spreadsheet editing tools',
    entry_points={
        'console_scripts': [
            'spreadsheet_wrangler=spreadsheet_wrangler.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=LONG_DESCRIPTION  + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='spreadsheet_wrangler',
    name='spreadsheet_wrangler',
    packages=find_packages(include=['spreadsheet_wrangler', 'spreadsheet_wrangler.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/snhobbs/spreadsheet-wrangler',
    version='0.1.5',
    zip_safe=False,
)
