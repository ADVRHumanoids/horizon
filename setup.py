import setuptools

from setuptools.command.develop import develop
from setuptools.command.build_py import build_py

import os
import codecs
import subprocess

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def _pre_build(dirname):
    # create a build dir and run 'make generate_python_package'
    current_dir = os.getcwd()
    print('building from dir: ', current_dir)
    os.makedirs(dirname, exist_ok=True)
    build_dir = current_dir + '/' + dirname

    try:
        p = subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", "../horizon/cpp"], cwd=build_dir)
    except subprocess.CalledProcessError:
        raise

    try:
        p = subprocess.run(["make", "-j8"], cwd=build_dir)
    except subprocess.CalledProcessError:
        raise

    try:
        p = subprocess.run(["make", "generate_python_package", "-j8"], cwd=build_dir)
    except subprocess.CalledProcessError:
        raise

class CustomBuild(build_py):
    # called by pip install and by python setup.py build and python setup.py install
    # build_py is not called by pip install -e
    def run(self):
        dir_name = 'temp_build'
        _pre_build(dir_name)
        build_py.run(self)

class CustomDevelop(develop):
    # called by pip install -e 
    def run(self):
        dir_name = 'temp_build'
        _pre_build(dir_name)
        develop.run(self)

setuptools.setup(
    name="casadi_horizon",
    version=get_version("horizon/__init__.py"),
    author="Francesco Ruscelli",
    author_email="francesco.ruscelli@iit.it",
    description="Library for Trajectory Optimization based on CasADi",
    long_description_content_type="text/markdown",
    url="https://github.com/ADVRHumanoids/horizon",
    packages=['horizon', 'horizon.utils', 'horizon.solvers', 'horizon.transcriptions', 'horizon.examples', 'horizon.ros', 'horizon.rhc'],
    install_requires=['casadi', 'numpy', 'matplotlib', 'scipy', 'casadi-kin-dyn', 'rospkg'],
    python_requires=">=3.6",
    cmdclass={'build_py': CustomBuild,
              'develop': CustomDevelop},
    ext_modules=[
        setuptools.Extension(
            name="ilqrext", sources=[]
        ),
        setuptools.Extension(
            name="sqpext", sources=[]
        )
    ]
)
