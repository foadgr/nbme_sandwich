from setuptools import setup, find_packages

setup(
    name="NBME Sandwich",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'kaggle',
        'scikit-learn',
        'spacy',
        'pygls',
        'transformers[torch]',
        'wandb'
    ]
)
