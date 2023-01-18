import sys
import os
from shutil import rmtree
from setuptools import find_packages, setup, Command

version_file = 'aimlflow/VERSION'

__version__ = None
with open(version_file) as vf:
    __version__ = vf.read().strip()


here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data
NAME = 'aim-mlflow'
DESCRIPTION = 'Aim-MLflow integration'
VERSION = __version__
REQUIRES_PYTHON = '>=3.7.0'

readme_file = 'README.md'
readme_text = open('/'.join((here, readme_file)), encoding="utf-8").read()
LONG_DESCRIPTION = readme_text.strip()

# Get packages
packages = find_packages(exclude=('tests', 'examples'))
version_files = ['../aimlflow/VERSION', ]

package_data = {'aimlflow': version_files}


REQUIRED = [
    'aim',
    'mlflow',
    'tqdm',
]


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    packages=packages,
    package_data=package_data,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    cmdclass={
        'upload': UploadCommand
    },
    entry_points={
        'console_scripts': [
            'aimlflow=aimlflow.cli:cli_entry_point',
        ],
    },
)


