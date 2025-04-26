from setuptools import setup, find_packages

setup(
    name='sdswrapper',
    version='0.2.6',
    description='A Python package for spatial data science workflows.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anderson Aires Eduardo',
    author_email='andersonaed@example.com',
    url='https://github.com/AndersonEduardo/sdswrapper',
    packages=find_packages(include=['sdswrapper', 'sdswrapper.*']),  # Inclui todos os submódulos
    include_package_data=True,  # Garante que arquivos adicionais sejam incluídos
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'pykrige'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)