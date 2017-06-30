from distutils.core import setup, Extension
from Cython.Build import cythonize
from glob import glob
from os.path import join

cpp_dir = join('fastText', 'src')

sources = ['pyfasttext.pyx', 'fasttext_access.cpp', 'exit.cpp']
# add all the fasttext source files except main.cc
sources.extend(set(glob(join(cpp_dir, '*.cc'))).difference({join(cpp_dir, 'main.cc')}))

setup(ext_modules = cythonize(Extension(
    "pyfasttext",
    sources=sources,
    include_dirs=['.'],
    libraries=['pthread'],
    extra_link_args=['-Wl,--wrap=exit'],
    language="c++",
)))
