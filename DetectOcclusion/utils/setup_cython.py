# code for compile a cython .pyx file
# usage: python setup_cython.py build_ext --inplace

import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension("edge_nms", ["edge_nms.pyx"])]
setup(ext_modules=cythonize(extensions), include_dirs=[numpy.get_include()])
