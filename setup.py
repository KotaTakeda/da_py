from setuptools import setup

setup(
    name='da_py',
    version='0.4.0',
    author='Kota Takeda',
    description='',
    long_description='',
    license='MIT',
    package_dir={'da': 'src'},
    install_requires=['numpy>=1.19', 'scipy>=1.1'],  # not strict
)
