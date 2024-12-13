# Machine Learning in Finance I

## Overview

This project aims to leverage machine learning techniques to address specific challenges and opportunities in the finance domain. 
The focus is on developing predictive models and data-driven insights that can enhance decision-making processes in financial applications.

## Table of Contents

- [Machine Learning in Finance I](#machine-learning-in-finance-i)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [For venv](#for-venv)
    - [For conda](#for-conda)
  - [Usage](#usage)
      - [Jupyter](#jupyter)
    - [JupyterLab](#jupyterlab)
    - [IPython](#ipython)
  - [How to run your Kedro pipeline](#how-to-run-your-kedro-pipeline)
  - [How to test your Kedro project](#how-to-test-your-kedro-project)
  - [Data](#data)
    - [Data Sources](#data-sources)
  - [Models](#models)

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository**:
   
```bash
git clone https://github.com/kwojdalski/ml_in_finance_i_project.git
cd ml_in_finance_i_project
```

2. **Install required packages**: It is recommended to create a virtual environment to manage dependencies. You can do this using venv or conda:

### For venv

```python
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### For conda

```python
conda create --name finance_ml python=3.12
conda activate finance_ml
conda install --file requirements.txt
```

Make sure you have Python and pip installed on your machine.

## Usage

To run the project, execute the main script or the appropriate Jupyter notebook:

Example: 
```python
python src/ml_in_finance_project/ml_qrt_project_final.py
```

or

Launch Jupyter Notebook:

Example:
```python
jupyter notebook
Open notebook.ipynb and run the cells.
Provide specific instructions on how to input data and interpret results if applicable.
```

Each file with model has its corresponding notebook. In order to convert .py to .ipynb, use the following command:

Example:
```python
jupytext --to ipynb src/ml_qrt_project_final.py --output notebooks/ml_qrt_project_final.ipynb
```

#### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```


## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


## Data

### Data Sources

Detail the datasets used in this project. If applicable, include sources like:

- Historical stock prices ([QRT Problem](https://www.quantrocket.com/qrt-problem/))


## Models

Model Selection
Outline the machine learning models used in this project, including:

- Linear Regression
- Random Forest
- Neural Networks
- Hyperparameter Tuning

Describe the methods used for hyperparameter tuning (e.g., Grid Search, Random Search).

