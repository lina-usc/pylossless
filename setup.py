# Authors: Christian O'Reilly <christian.oreilly@sc.edu>;
# Scott Huberty <scott.huberty@mail.mcgill.ca>
# James Desjardins <jim.a.desjardins@gmail.com>
# License: MIT

from setuptools import setup

install_requires = ['numpy', 'EDFlib', 'mne', 'mne_bids', 'pandas', 
                    'xarray', 'scipy', 'mne_icalabel', 'pyyaml', 
                    'IProgress', 'ipywidgets', 'scikit-learn']

setup(
    name='pylossless',
    version="0.0.1",
    description='Python port of EEG-IP-L pipeline for preprocessing EEG.',
    python_requires='>=3.5',
    author="Scott Huberty",
    author_email='seh33@uw.edu',
    url='https://github.com/scott-huberty/pylossless',
    packages=['pylossless'],
    install_requires=install_requires,
    include_package_data=True,
    entry_points={"console_scripts": ["pylossless_qc=pylossless.dash.app:main"]})
