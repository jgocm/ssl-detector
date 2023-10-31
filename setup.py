'''
#       Install Project Requirements 
'''
from setuptools import setup, find_packages

setup(name='ssl-detector',
    url="https://github.com/jgocm/ssl-detector",
    description="Object Detection and Localization for RoboCup Small Size League (SSL)",
    packages=[package for package in find_packages() if package.startswith("ssl-detector")]
)

# TODO: add install_requires=['gym==0.21.0', 'rc-robosim>=1.2.0', 'pyglet', 'protobuf']