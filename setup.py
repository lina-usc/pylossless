# Authors: Christian O'Reilly <christian.oreilly@gmail.com>;
# Scott Huberty <scott.huberty@mail.mcgill.ca>
# James Desjardins <jim.a.desjardins@gmail.com>
# License: MIT

from setuptools import setup


if __name__ == "__main__":
    hard_dependencies = ('numpy', 'scipy', 'mne', 'mne-bids','pandas', 'xarray','pyaml')
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            for hard_dep in hard_dependencies:
                if req.startswith(hard_dep):
                    install_requires.append(req)

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
