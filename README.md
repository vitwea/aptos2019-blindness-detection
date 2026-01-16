APTOS 2019 Blindness Detection

<p align="center"> <a href="#project-overview"><img src="https://img.shields.io/badge/-Project%20Overview-blue" alt="Project Overview" /></a>    <a href="#project-structure"><img src="https://img.shields.io/badge/-Project%20Structure-blue" alt="Project Structure" /></a>    <a href="#installation--setup"><img src="https://img.shields.io/badge/-Installation%20%26%20Setup-blue" alt="Installation & Setup" /></a>    <a href="#dataset"><img src="https://img.shields.io/badge/-Dataset-blue" alt="Dataset" /></a>    <a href="#exploratory-data-analysis"><img src="https://img.shields.io/badge/-Exploratory%20Data%20Analysis-blue" alt="Exploratory Data Analysis" /></a>    <a href="#baseline-model"><img src="https://img.shields.io/badge/-Baseline%20Model-blue" alt="Baseline Model" /></a>    <a href="#data-preprocessing--augmentation"><img src="https://img.shields.io/badge/-Data%20Preprocessing%20%26%20Augmentation-blue" alt="Data Preprocessing & Augmentation" /></a>    <a href="#training"><img src="https://img.shields.io/badge/-Training-blue" alt="Training" /></a>    <a href="#evaluation"><img src="https://img.shields.io/badge/-Evaluation-blue" alt="Evaluation" /></a>    <a href="#results-baseline"><img src="https://img.shields.io/badge/-Results%20(Baseline)-blue" alt="Results Baseline" /></a>    <a href="#next-steps"><img src="https://img.shields.io/badge/-Next%20Steps-blue" alt="Next Steps" /></a>    <a href="#license"><img src="https://img.shields.io/badge/-License-blue" alt="License" /></a>    <a href="#author"><img src="https://img.shields.io/badge/-Author-blue" alt="Author" /></a> </p>

Project Overview

    This repository contains a clean and reproducible end-to-end solution for the APTOS 2019 Blindness Detection Kaggle competition. The goal of this project is to practice professional machine learning workflows, including:

    Reproducible environment setup

    Modular project structure

    Data loading and preprocessing

    Baseline deep learning model (ResNet18)

    Training and evaluation notebooks

    Metrics, confusion matrix and error analysis

    Version control with Git and GitHub

Project Structure

    aptos2019-blindness-detection/
    ├── data/                # Raw and processed data (ignored by Git)
    │   ├── raw/
    │   └── processed/
    ├── notebooks/           # Jupyter notebooks
    │   ├── 01_exploration.ipynb
    │   ├── 02_training.ipynb
    │   └── 03_evaluation.ipynb
    ├── src/                 # Reusable Python modules
    │   ├── data_loading.py
    │   ├── preprocessing.py
    │   ├── train.py
    │   └── metrics.py
    ├── models/              # Saved model weights
    ├── requirements.txt
    ├── .gitignore
    └── README.md

Installation & Setup

    1. Create and activate a virtual environment

        python -m venv .venv
        .venv\Scripts\Activate.ps1   # Windows

    2. Install dependencies

        pip install -r requirements.txt

Dataset

    The dataset comes from the APTOS 2019 Blindness Detection competition on Kaggle.

    Download it using the Kaggle CLI:

        kaggle competitions download -c aptos2019-blindness-detection -p data/raw

    Extract the files:

        Expand-Archive data/raw/aptos2019-blindness-detection.zip -DestinationPath data/raw


Exploratory Data Analysis

    The notebook 01_exploration.ipynb includes:

    Class distribution analysis

    Sample image visualization

    Basic dataset inspection

This step validates that the dataset is correctly loaded and helps understand the problem.


Baseline Model

    The baseline model is a ResNet18 pretrained on ImageNet, adapted to the 5-class diabetic retinopathy classification task.

    Implemented in:

        src/train.py

    Includes:

        Training loop

        Evaluation loop

        Loss tracking

        Model saving


Data Preprocessing & Augmentation

    Implemented in:

        src/preprocessing.py

    Includes:

    Basic transforms

        Augmentations:

            Random horizontal flip

            Random rotation

            Color jitter

            Resize + ToTensor

    These improve generalization and robustness.


Training

    Run the training notebook:

        notebooks/02_training.ipynb

    It includes:

        Model initialization

        Training loop

        Loss curve visualization

        Saving model weights


Evaluation

    The evaluation notebook:

        notebooks/03_evaluation.ipynb

    Provides:

        Predictions on the dataset

        Confusion matrix

        Classification report (precision, recall, F1)

        Error analysis

    Metrics implemented in:

        src/metrics.py


Results (Baseline)

    Example baseline results (may vary):

        Training loss decreases over epochs

        Confusion matrix shows class imbalance impact

        F1-scores reflect difficulty of minority classes

    This baseline serves as a foundation for future improvements.


Next Steps

    Potential improvements:

        Advanced augmentations

        Balanced sampling

        Learning rate scheduling

        EfficientNet or ViT models

        Stratified train/validation split

        Hyperparameter tuning

        Better loss functions (e.g., focal loss)


License

    This project is for educational and portfolio purposes.


Author

    Pablo Monclús RadigalesMachine Learning & Data Engineering EnthusiastZaragoza, Spain