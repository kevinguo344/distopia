from setuptools import setup, find_packages
import distopia

with open('README.md') as fh:
    long_description = fh.read()

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
    install_requires=['pytest', 'scipy', 'pyshp', 'numpy'],
    package_data={'distopia': ['data/*', ]},
)
