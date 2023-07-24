#!/bin/bash

# Create the conda environment
conda create -n nbme_sandwich python=3.11 -y

# Activate the conda environment
source activate nbme_sandwich

# Install the required packages
pip install --editable .

# Download Kaggle competition

kaggle competitions download -c nbme-score-clinical-patient-notes

# Unzip the data
mkdir data && unzip nbme-score-clinical-patient-notes.zip ./data