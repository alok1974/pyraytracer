# -*- coding: utf-8 -*-
from distutils.core import setup
from glob import glob


PACKAGE_NAME = 'pyraytracer'
PACKAGE_VERSION = '0.0.0'


setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description='A raytrace engine written in python',
    author='Alok Gandhi',
    author_email='alok.gandhi2002@gmail.com',
    url='https://github.com/alok1974/pyraytracer',
    packages=[
        'pyraytracer',
    ],
    package_data={
        'pyraytracer': [],
    },
    package_dir={
        'pyraytracer': 'src/pyraytracer'
    },
    download_url=(
        'https://github.com/alok1974/pyraytracer/archive/'
        f'v{PACKAGE_VERSION}.tar.gz'),
    scripts=glob('src/scripts/*'),
    install_requires=[
        'numpy >= 1.25.2',
        'Pillow >= 10.0.0',
        'pydantic_core >= 2.6.1',
        'pytest >= 7.4.1',
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language:: Python:: 3.7',
        'Topic :: Games/Entertainment :: Board Games',
    ],
)
