# ml_in_finance_i_project

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project with Kedro-Viz and PySpark setup, which was generated using `kedro 0.19.10`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Installation

### Install dependencies:

```
pip install -r requirements.txt
```

### Download data:

* Go to: https://challengedata.ens.fr/participants/challenges/23/
* Register and download the data
* Rename files from `x_train_*`, `y_train_*`, `x_test_*` to `x_train.csv`, `y_train.csv`, `x_test.csv`
* Put them in `data/01_raw/`

### Run the pipeline:

1) Run the following command to run all the pipelines:

```
kedro run
```

2) Run specific pipeline:

```
kedro run --pipeline=reporting
```

3) Run specific node:

```
kedro run --node=plot_returns_volume_node
```

In order to run visualization, run the following command:

```
kedro viz run
```


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter

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
