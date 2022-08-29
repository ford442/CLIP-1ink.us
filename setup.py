import os
from distutils.extension import Extension;
import pkg_resources
from setuptools import setup, find_packages
extensions = [Extension('clip',['clip/clip.py'])];
from Cython.Build import cythonize

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
    ext_modules = cythonize(extensions,nthreads=4,compiler_directives={'infer_types':True}),
)
