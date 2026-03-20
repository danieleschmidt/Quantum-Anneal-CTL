from setuptools import setup, find_packages

setup(
    name="quantum-anneal-ctl",
    version="0.1.0",
    description="Quantum annealing-inspired HVAC control for building microgrids",
    author="Daniel Schmidt",
    author_email="danschmidt88@gmail.com",
    url="https://github.com/danieleschmidt/Quantum-Anneal-CTL",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
