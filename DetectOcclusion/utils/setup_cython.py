# code for compile a cython .pyx file
# usage: python setup_cython.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(name='edge nms app', ext_modules=cythonize("edge_nms.pyx"))
