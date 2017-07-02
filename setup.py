from distutils.core import setup, Extension
from Cython.Build import cythonize
from glob import glob
from os.path import join
import os

cpp_dir = join('src', 'fastText', 'src')

sources = ['src/pyfasttext.pyx', 'src/fasttext_access.cpp']
# add all the fasttext source files except main.cc
sources.extend(set(glob(join(cpp_dir, '*.cc'))).difference({join(cpp_dir, 'main.cc')}))

# exit() replacement does not work when we use extra_compile_args
os.environ['CFLAGS'] = '-iquote . -include src/custom_exit.h'

setup(ext_modules = cythonize(Extension(
    'pyfasttext',
    sources=sources,
    libraries=['pthread'],
    include_dirs=['.'],
    language='c++',
)))
