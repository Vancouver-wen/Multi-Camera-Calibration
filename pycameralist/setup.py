#!/usr/bin/env python
# coding=utf-8


from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from PyCameraList import __version__
# import eigency

extensions = [
    Extension("PyCameraList.camera_device", ["PyCameraList\camera_device.pyx"]),
]

LIB_NAMES = ['quartz', 'strmiids', "kernel32", "user32", "gdi32", "winspool", "comdlg32", "advapi32", "shell32",
             "ole32", "oleaut32", "uuid", "odbc32", "odbccp32", "comsuppw"]
LIB_DIRS = []  # C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64
for ext in extensions:
    ext.include_dirs = ["."] # + eigency.get_includes()
    ext.libraries = LIB_NAMES
    ext.library_dirs = LIB_DIRS
    ext.language = "c++"
    ext.cython_directives = {'language_level': "3"}  # all are Python-3
    ext.define_macros = [("_ENABLE_EXTENDED_ALIGNED_STORAGE", None)]

setup(
    name='PyCameraList',
    version=__version__,
    author='Peter Jiang',
    email='07jiangbin@163.com',
    description='List all camera/audio sources for windows users.',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    python_requires='>=3.6',
)
