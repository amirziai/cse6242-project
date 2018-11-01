from setuptools import setup


setup(
    name='cse6242-project',
    versioning='build-id',
    author='Team 14',
    author_email='team14-not-really@gatech.edu',
    keywords='clustering',
    url='https://github.com/amirziai/cse6242-project',
    setup_requires=['setupmeta'],
    python_requires='>=3.6',
    install_requires=[
        'pytest',
        'pandas',
        'scipy',
        'scikit-learn',
        'nltk',
        'flask',
        'gunicorn',
    ],
    extras_require={
        'test': ['tox'],
    },
)
