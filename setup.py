# fmt: off
from setuptools import setup

setup(
    name="Gym-MinAtar",
    version="1.0.0",
    license="CC-BY-4.0",
    author="Simone Parisi",
    packages=["gym_minatar"],
    python_requires=">=3.9",
    install_requires=["gymnasium", "pygame", "numpy"],
    extras_require={
        "playground": ["pynput", "imageio"],
    },
)
