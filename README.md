<img heigth="8" src="https://i.imgur.com/5VTYUPE.png" alt="banner">

<h1 align="center">💳 Credit Card Fraud Detection 🚨</h1>

<p align="center">An extremely rare event classification system for detecting credit card fraud using machine learning with active learning strategies.</p>

<p align="center">
  <a href="https://joefavergel.github.io/">joefavergel.github.io</a>
  <br> <br>
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#life-cycle">Life Cycle</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#license">License</a>
  <br> <br>
  <a target="_blank">
    <img src="https://github.com/QData/TextAttack/workflows/Github%20PyTest/badge.svg" alt="Github Runner Covergae Status">
  </a>
  <a href="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000">
    <img src="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000" alt="Version" height="18">
  </a>
  <a href="https://www.linkedin.com/in/joseph-fabricio-vergel-becerra/" target="_blank">
    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://twitter.com/joefavergel" target="_blank">
    <img alt="Twitter: joefavergel" src="https://img.shields.io/twitter/follow/joefavergel.svg?style=social"/>
  </a>
</p>


---

## About

This project addresses **credit card fraud detection** as an **extremely rare event classification** problem. Using a highly imbalanced dataset where fraudulent transactions represent only 0.172% of all cases, the project implements preprocessing techniques and machine learning models properly for **rare event classification**, and active learning strategies to achieve high sensitivity without sacrificing precision.

### Key Highlights

- **Dataset**: Credit Card Fraud Detection from Kaggle (284,807 transactions, 492 frauds)
- **Challenge**: Handling extreme class imbalance (99.83% legitimate vs 0.17% fraudulent)
- **Approach**:
  - Feature engineering: Temporal features from transaction sequences
  - Robust preprocessing: PowerTransformer (Box-Cox) for skewed distributions
  - Baseline models: OneClassSVM (anomaly detection) and Logistic Regression
  - Ensemble models: Custom XGBoost implementation with automatic class weight balancing
  - Threshold optimization: Dual strategies (maximize F1 vs. guarantee minimum recall)
  - Active Learning with Uncertainty Sampling for efficient label acquisition
- **Results**:
  - XGBoost significantly outperforms baseline models across all metrics
  - High fraud detection rates (recall ≥85%) with controlled false positive rates
  - Demonstrated effectiveness of temporal feature engineering and threshold optimization

The complete analysis and implementation can be found in the notebook `notebooks/extremely_rare_event_classification.ipynb`.


---

## Features

This project is built on `Python 3.12` with:
- **Data Processing**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/), [scikit-activeml](https://scikit-activeml.github.io/scikit-activeml-docs/)
- **Visualization**: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)

For development, the library use:

- Dependency mangament with [uv](https://docs.astral.sh/uv/)
- Formatting, import sorting and linting with [ruff](https://docs.astral.sh/ruff/)
- Git hooks that run all the above with [pre-commit](https://pre-commit.com/)
- Testing with [pytest](https://docs.pytest.org/en/latest/)


---

## Project Structure

The main analysis is contained in the notebook `notebooks/extremely_rare_event_classification.ipynb` and follows this workflow:

1. **Data Acquisition**: Download the Credit Card Fraud Detection dataset from Kaggle
2. **Exploratory Data Analysis**:
   - Examine class distribution (99.83% normal, 0.17% fraudulent)
   - Analyze feature distributions and correlations
   - Identify key patterns in fraudulent transactions
3. **Preprocessing**:
   - Feature engineering: Transform `Time` into meaningful temporal features
   - Robust scaling using `PowerTransformer` (Box-Cox) for skewed distributions
   - Stratified train-test split to maintain class proportions
4. **Model Training**:
   - **Baseline Models**:
     - OneClassSVM (unsupervised anomaly detection)
     - Logistic Regression with class balancing
   - **Ensemble Models**:
     - `XGBoostClassifierImbalanced` class with automatic `scale_pos_weight` calculation
     - Early stopping and best model restoration
     - Feature importance analysis
   - **Threshold Optimization**:
     - Strategy 1: Maximize F1-Score
     - Strategy 2: Guarantee minimum recall (≥85%)
   - **Active Learning**: Uncertainty Sampling with least-confident query strategy for iterative performance improvement
5. **Evaluation**:
   - Comprehensive metrics: ROC-AUC, PR-AUC, F1-score, recall, precision
   - Confusion matrices and performance comparisons
   - Model comparison: Logistic Regression vs. XGBoost

---

## Life Cycle

As a proposal for the data science life cycle, [OSEMN](https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492) is mainly proposed. Standing for Obtain, Scrub, Explore, Model, and iNterpret, OSEMN is a five-phase life cycle.

Other good option is [Microsoft TDSP: The Team Data Science Process](https://learn.microsoft.com/en-us/azure/architecture/data-science-process/overview) combines many modern agile practices with the life cycle. It has five steps: Business Understanding, Data Acquisition and Understanding, Modeling, Deployment, and Customer Acceptance.

The important thing is that if you think they should be combined and form their own life cycle, feel free to do so.


---

## Contribute

First, make sure that before enabling pipenv, you must have `Python 3.12` installed. If it does not correspond to the version you have installed, you can create a conda environment with:

```sh
# Create and activate python 3.12 virutal environment
$ conda create -n py312 python=3.12
$ conda activate py312
```

Now, you can managament the project dependencies with `uv`. To create de virtual environment and install all dependencies follow:

```sh
# Install uv using pip
$ pip install uv

# Install development dependencies
$ uv sync --all-packages

# Activate virtual environment
$ source .venv/bin/activate
```

Once the dependencies are installed, we need to notify `Jupyter` of this new `Python` environment by creating a kernel:

```sh
$ ipython kernel install --user --name KERNEL_NAME
```

Finally, before making any changes to the library, be sure to review the [GitFlow](https://www.atlassian.com/es/git/tutorials/comparing-workflows/gitflow-workflow) guide and make any changes outside of the `master` branch.
