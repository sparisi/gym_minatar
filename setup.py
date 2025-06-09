# fmt: off
from setuptools import setup

setup(
    name="Gym-MinAtar",
    version="1.0",
    license="CC-BY-4.0",
    author="Simone Parisi",
    packages=["gym_minatar"],
    install_requires=["gymnasium", "pygame", "pynput"],
)
