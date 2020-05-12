from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.3.1',
                    'h5py==2.10.0']

setup(
    name='trainer',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras dog breed classifier module, with normalization.'
)
