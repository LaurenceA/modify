from setuptools import setup

setup(
    name="modify",
    version="0.1.0",
    packages=['modify'],
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
    ],
    python_requires='>=3.6',
)
