""" Authors:
Christian O'Reilly <christian.oreilly@sc.edu>
Scott Huberty <seh33@uw.edu>
James Desjardins <jim.a.desjardins@gmail.com>
Tyler Collins <collins.tyler.k@gmail.com>
License: MIT
"""

from pathlib import Path
from setuptools import setup, find_packages

with Path('requirements.txt').open() as f:
    requirements = f.read().splitlines()

extras = {
    'dash': 'requirements_qc.txt',
    'test': 'requirements_testing.txt',
    'doc': './docs/requirements_doc.txt'
}

extras_require = {}
for extra, req_file in extras.items():
    with Path(req_file).open() as file:
        requirements_extra = file.read().splitlines()
    extras_require[extra] = requirements_extra

qc_entry_point = ["pylossless_qc=pylossless.dash.pylossless_qcr:main"]
setup(
    name='pylossless',
    version='0.0.1',
    description='Python port of EEG-IP-L pipeline for preprocessing EEG.',
    author="Scott Huberty",
    author_email='seh33@uw.edu',
    url='https://github.com/lina-usc/pylossless',
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    entry_points={"console_scripts": qc_entry_point}
)
