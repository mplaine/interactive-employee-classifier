![Interactive Employee Classifier](./static/screenshot.png)

# Interactive Employee Classifier

Want to predict employee attrition using the features from a HR dataset? Well, look no further! This interactive tool allows you to build and evaluate different machine learning classification models exactly for this purpose.


## Dataset

This project uses preprocessed versions of the [HR dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction) from Kaggle. The original dataset contains 14,999 rows and 10 columns, each row representing self-reported information from employees.


## Quick Start Guide

### Installation

#### Prerequisites

* [Python 3.11+](https://www.python.org/)

#### Virtual Environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), after which we can create a dedicated Python virtual environment for the project:

```bash
# Create a virtual environment for Python 3.11
conda create -n interactive-employee-classifier python=3.11

# Activate the environment
conda activate interactive-employee-classifier

# Deactivate the environment, if needed
conda deactivate
```

#### Python Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Make sure the Python virtual environment is active before installing requirements.


### Usage

Run the Streamlit application:

```bash
streamlit run app.py
```


## License

Copyright (c) 2024 [Markku Laine](https://markkulaine.com)

This software is distributed under the terms of the [MIT License](https://opensource.org/license/mit/). See [LICENSE](./LICENSE) for details.
