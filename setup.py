from setuptools import setup, Extension
from Cython.Build import cythonize
from glob import glob
from os.path import join
import os
import sys

def to_bool(val):
    if not val:
        val = 0
    else:
        try:
            val = int(val)
        except:
            val = 1
    return bool(val)

# numpy support is optional
USE_NUMPY = to_bool(os.environ.get('USE_NUMPY', '1'))

include_dirs = ['.']
install_requires = ['future']

if USE_NUMPY:
    import numpy as np
    include_dirs.append(np.get_include())
    install_requires.append('numpy')

cpp_dir = join('src', 'fastText', 'src')

sources = ['src/pyfasttext.pyx', 'src/fasttext_access.cpp']
# add all the fasttext source files except main.cc
sources.extend(set(glob(join(cpp_dir, '*.cc'))).difference({join(cpp_dir, 'main.cc')}))

# exit() replacement does not work when we use extra_compile_args
os.environ['CFLAGS'] = '-iquote . -include src/custom_exit.h'

extension = Extension(
    'pyfasttext',
    sources=sources,
    libraries=['pthread'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=['-std=c++0x', '-Wno-sign-compare'])

setup(name='pyfasttext',
      version='0.2.0',
      author='Vincent Rasneur',
      author_email='vrasneur@free.fr',
      url='https://github.com/vrasneur/pyfasttext',
      download_url='https://github.com/vrasneur/pyfasttext/releases/download/0.2.0/pyfasttext-0.2.0.tar.gz',
      description='Yet another Python binding for fastText',
      long_description=open('README.rst', 'r').read(),
      license='GPLv3',
      package_dir={'': 'src'},
      ext_modules=cythonize(extension, compile_time_env={'USE_NUMPY': USE_NUMPY}),
      install_requires=install_requires,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
