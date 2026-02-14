from setuptools import setup

setup(
    name="da_py",
    version="0.5.1",
    author="Kota Takeda",
    description="",
    long_description="",
    license="MIT",
    package_dir={"da": "src"},
    packages=["da"],
    install_requires=["numpy>=1.19", "scipy>=1.1"],  # not strict
)
