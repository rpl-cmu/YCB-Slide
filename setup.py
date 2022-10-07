
#!/usr/bin/env python

from setuptools import setup
import shlex
import subprocess

def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode()

version = git_version()
setup(
    name='ycb_slide',
    version=version,
    author='Sudharshan Suresh',
    author_email='suddhus@gmail.com',
    packages=['ycb_slide'], 
)
