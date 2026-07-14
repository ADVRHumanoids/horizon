import setuptools

from setuptools.command.develop import develop
from setuptools.command.build_py import build_py

import os
import codecs
import subprocess
import site
import sys

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

def _run_build_step(number, total, description, command):
    separator = "=" * 72
    print("\n" + separator, flush=True)
    print(
        "[Horizon C++] Step {}/{}: {}".format(number, total, description),
        flush=True,
    )
    print("[Horizon C++] Command: {}".format(" ".join(command)), flush=True)
    print(separator, flush=True)
    subprocess.run(command, check=True)
    print("[Horizon C++] Step {}/{} completed".format(number, total), flush=True)

def _pre_build(dirname):
    # pip/Forest may invoke setup.py from a temporary working directory, so
    # resolve every path from this file rather than from os.getcwd().
    source_dir = os.path.abspath(os.path.dirname(__file__))
    cpp_dir = os.path.join(source_dir, "horizon", "cpp")
    build_dir = os.path.join(source_dir, dirname)
    default_prefix = (
        sys.prefix
        if sys.prefix != sys.base_prefix or os.access(sys.prefix, os.W_OK)
        else site.USER_BASE
    )
    install_prefix = os.environ.get("CMAKE_INSTALL_PREFIX", default_prefix)
    bundle_external_libs = os.environ.get(
        "HORIZON_BUNDLE_EXTERNAL_LIBS", "0"
    ).lower() in {"1", "true", "yes", "on"}
    total_steps = 4 if bundle_external_libs else 3

    print("\n[Horizon C++] Starting native build", flush=True)
    print("[Horizon C++] Source:         {}".format(cpp_dir), flush=True)
    print("[Horizon C++] Build tree:     {}".format(build_dir), flush=True)
    print("[Horizon C++] Install prefix: {}".format(install_prefix), flush=True)
    os.makedirs(build_dir, exist_ok=True)

    _run_build_step(
        1,
        total_steps,
        "configure the C++ project",
        [
            "cmake",
            "-S", cpp_dir,
            "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        ],
    )
    _run_build_step(
        2,
        total_steps,
        "compile the C++ libraries and Python modules",
        [
            "cmake", "--build", build_dir,
            "--target", "pyilqr", "pysqp",
            "--parallel", "8",
        ],
    )
    if bundle_external_libs:
        _run_build_step(
            3,
            total_steps,
            "bundle external libraries for binary distribution",
            [
                "cmake", "--build", build_dir,
                "--target", "generate_python_package",
                "--parallel", "8",
            ],
        )
    else:
        print(
            "[Horizon C++] External-library bundling skipped "
            "(editable installs use the current environment)",
            flush=True,
        )
    _run_build_step(
        total_steps,
        total_steps,
        "install C++ headers, libraries, and CMake package files",
        ["cmake", "--install", build_dir],
    )
    print(
        "\n[Horizon C++] Build and installation completed successfully",
        flush=True,
    )

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

class BinaryDistribution(setuptools.Distribution):
    """Mark wheels as platform-specific without creating fake extensions."""
    def has_ext_modules(self):
        return True

setuptools.setup(
    name="casadi_horizon",
    version=get_version("horizon/__init__.py"),
    author="Francesco Ruscelli",
    author_email="francesco.ruscelli@iit.it",
    description="Library for Trajectory Optimization based on CasADi",
    long_description_content_type="text/markdown",
    url="https://github.com/ADVRHumanoids/horizon",
    packages=setuptools.find_packages(
        include=["horizon", "horizon.*"],
        exclude=["horizon.cpp", "horizon.cpp.*"],
    ),
    include_package_data=False,
    package_data={
        "horizon.solvers": ["*.so"],
        "horizon": ["external_libs/*"],
        "horizon.examples": ["**/*.launch", "**/*.rviz", "**/*.urdf", "**/*.xml"],
    },
    install_requires=['numpy', 'matplotlib', 'scipy', 'casadi-kin-dyn'],
    python_requires=">=3.6",
    cmdclass={'build_py': CustomBuild,
              'develop': CustomDevelop},
    distclass=BinaryDistribution,
)
