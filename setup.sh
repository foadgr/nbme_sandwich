#!/bin/bash

# Create the conda environment
conda create -n ner_nbme python=3.11 -y

# Activate the conda environment
conda activate ner_nbme

# Install the required packages
pip install --upgrade --editable .

# Add conda environment to Jupyter
python -m ipykernel install --user --name=ner_nbme

# Update Jupyter and ipywidgets
pip install --upgrade jupyter ipywidgets

#Download `en_core_web_sm` SpaCy model
python -m spacy download en_core_web_sm

# Download Kaggle competition

kaggle competitions download -c nbme-score-clinical-patient-notes

# Unzip the data
mkdir data && unzip nbme-score-clinical-patient-notes.zip ./data