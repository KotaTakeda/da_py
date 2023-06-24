from setuptools import setup

setup(
    name='da_py',
    version='0.1.0',
    author='Kota Takeda',
    description='',
    long_description='',
    license='MIT',
    package_dir={'da_py': 'src'},
    install_requires=['numpy>=1.19', 'scipy>=1.1'],  # not strict
)
