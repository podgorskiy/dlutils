#
# Copyright 2019 Stanislav Pidhorskyi. All rights reserved.
# License: https://raw.githubusercontent.com/podgorskiy/bimpy/master/LICENSE.txt
#

from setuptools import setup

from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dlutils',

    version='0.0.1',

    description='dlutils - collection of boilerplate code, usefull primitives, helpers.',
    long_description=long_description,

    url='https://github.com/podgorskiy/dlutils',

    author='Stanislav Pidhorskyi',
    author_email='stanislav@podgorskiy.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='deep-learning extension cpp c pytorch',

    packages=['dlutils'],
)
