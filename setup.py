from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include

ext = cythonize(Extension("TDMA", sources = ['TDMA.pyx'], language="c++"))
ext += cythonize(Extension("Cheb", sources = ['Cheb.pyx']))
ext += cythonize(Extension("PcorrSolve", sources = ['PcorrSolve.pyx']))

[e.extra_compile_args.extend(["-O3"]) for e in ext]
[e.include_dirs.extend([get_include()]) for e in ext]

setup(
    name = "SFTc",
    ext_modules = ext,
    py_modules = ["SFTc.py"],
    cmdclass = {'build_ext': build_ext})
