from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from glob import glob
from os.path import join
from subprocess import call
import os
import sys

VERSION = '0.4.5'

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
# cysignals support is optional too
USE_CYSIGNALS = to_bool(os.environ.get('USE_CYSIGNALS', '1'))
if sys.platform == 'win32':
    USE_CYSIGNALS = False

# if the module is being installed from pip using bdist_wheel or egg_info
# make sure cysignals is installed before compiling
if USE_CYSIGNALS and 'bdist_wheel' in sys.argv or 'egg_info' in sys.argv:
    try:
        import cysignals
    except ImportError:
        ret = call([sys.executable, '-m', 'pip', 'install', 'cysignals'])
        if ret:
            raise RuntimeError('cannot install cysignals with pip')

def get_fasttext_commit_hash():
    try:
        with open('.git/modules/fastText/HEAD', 'r') as f:
            return f.read().strip()
    except:
        return 'unknown'

include_dirs = ['.', 'src/variant/include', 'src']
setup_requires = []
install_requires = ['future', 'cysignals']

if USE_NUMPY:
    setup_requires.append('numpy')
    install_requires.append('numpy')

cpp_dir = join('src', 'fastText', 'src')

sources = ['src/pyfasttext.pyx', 'src/fasttext_access.cpp']
# add all the fasttext source files except main.cc
sources.extend(set(glob(join(cpp_dir, '*.cc'))).difference(set([join(cpp_dir, 'main.cc')])))

# exit() replacement does not work when we use extra_compile_args
os.environ['CFLAGS'] = '-iquote . -include src/custom_exit.h'

class BuildExt(build_ext):
    def build_extensions(self):
        extra_compile_args = self.extensions[0].extra_compile_args
        if 'clang' in self.compiler.compiler[0]:
            extra_compile_args.append('-std=c++1z')
        else:
            extra_compile_args.append('-std=c++0x')
        build_ext.build_extensions(self)

    def finalize_options(self):
        build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process
        if USE_NUMPY:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy as np
            self.include_dirs.append(np.get_include())

extension = Extension(
    'pyfasttext',
    sources=sources,
    libraries=['pthread'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=['-Wno-sign-compare'])

setup(name='pyfasttext',
      version=VERSION,
      author='Vincent Rasneur',
      author_email='vrasneur@free.fr',
      url='https://github.com/vrasneur/pyfasttext',
      download_url='https://github.com/vrasneur/pyfasttext/releases/download/%s/pyfasttext-%s.tar.gz' % (VERSION, VERSION),
      description='Yet another Python binding for fastText',
      long_description=open('README.rst', 'r').read(),
      license='GPLv3',
      package_dir={'': 'src'},
      ext_modules=cythonize([extension], compile_time_env={'USE_NUMPY': USE_NUMPY,
                                                           'USE_CYSIGNALS': USE_CYSIGNALS,
                                                           'VERSION': VERSION,
                                                           'FASTTEXT_VERSION': get_fasttext_commit_hash()}),
      setup_requires=setup_requires,
      install_requires=install_requires,
      cmdclass={'build_ext': BuildExt},
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: POSIX',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
