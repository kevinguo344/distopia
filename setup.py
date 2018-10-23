from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import distopia

with open('README.md') as fh:
    long_description = fh.read()


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed.
    https://stackoverflow.com/questions/2379898/make-distutils-look-for-
    numpy-header-files-in-the-correct-place
    """

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


setup(
    name='Distopia',
    version=distopia.__version__,
    author='hrc2da',
    url='http://hrc2da.github.io/distopia/',
    license='MIT',
    description='Voting Districts UI Architecture.',
    long_description=long_description,
    classifiers=['License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Intended Audience :: Developers'],
    packages=find_packages(),
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=cythonize([Extension(
        "distopia.mapping._voronoi", ["distopia/mapping/_voronoi.pyx"])]),
    install_requires=['pytest', 'scipy', 'pyshp', 'numpy', 'pyproj', 'oscpy',
                      'matplotlib', 'Cython', 'roslibpy'],
    package_data={'distopia': ['data/*', ]},
)
