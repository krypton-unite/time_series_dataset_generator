from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup(
    name='time_series_dataset_generator',
    version='0.1.10',
    description='Generator for time-series-dataset.',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='The Unlicense',
    packages=find_packages(exclude=("tests",)),
    author='Daniel Kaminski de Souza',
    author_email='daniel@kryptonunite.com',
    keywords=['Time series dataset generator'],
    url='https://github.com/krypton-unite/time_series_dataset_generator.git',
    download_url='https://pypi.org/project/time-series-dataset-generator/',
    python_requires=">=3.7",
    install_requires = [
        'time-series-dataset',
        'time-series-generator',
        'pandas',
        'sklearn',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'seaborn',
            'time_series_predictor',
            'time-series-models',
        ],
        'dev': [
            'bumpversion',
            'twine',
            'wheel',
            'pylint',
            'autopep8'
        ],
        'docs': [        
        ]
    }
)