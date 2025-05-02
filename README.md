# FPSO Failure Forecasting Case Study ⚙️📉

This project aims to anticipate equipment failures by analyzing patterns in sensor data and operational presets, using data exploration and predictive modeling.


Key technologies used include:
- **Pandas & NumPy** for data manipulation and analysis.
- **Scikit-learn** and **LightGBM** for model training and evaluation.
- **Matplotlib & Seaborn** for data visualization and exploration.

Final outputs include:
- A structured and cleaned dataset enriched with engineered features.
- Visual analysis of failure triggers and sensor behavior.
- A predictive model for failures.

---

## Project Structure
FPSO-failure-prediction/
├── data/
│ └── O_G_Equipment_Data.xlsx
│
├── notebooks/
│ └── FPSO_failure_prediction.ipynb
│
├── presentation/
│ └── FPSO_failure_prediction.html
│
├── src/
│ ├── init.py
│ ├── data_exploration.py
│ ├── data_preprocessing.py
│ ├── model_evaluate.py
│ └── model_train.py
│
├── .gitignore
├── setup.py
├── requirements.txt
└── README.md


## Installation

1. Clone repository:
    ```bash
    git clone https://github.com/YOUR-USERNAME/FPSO-Failure-Prediction.git
    cd FPSO-Failure-Prediction
    ```

2. Create virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Install directory as package in editable mode:
    ```bash
    pip install -e .
    ```
---

## Usage

You can see the whole procject in hmlr or execute a simple step workflow organized in a Jupyter notebook:

- **Notebook**: `FPSO_failure_prediction.ipynb`
- **Description**:
  - Clean and inspect sensor and preset data.
  - Identify which variables behaves precede failure.
  - Prdict failures using machine learning model
  - evaluate model performance
---

## Code Overview

| File                     | Description |
|--------------------------|-------------|
| `data_exploration.py` | Prepare the raw source to a model-ready dataset. |
| `data_preprocessing.py` | Provides various data analysis and visualizations for exploring failure patterns in the dataset. |
| `model_evaluate.py` | Evaluates model predictions with detailed metrics and visualizations. |
| `model_train.py` | ETrains and tunes LightGBM models using a full pipeline (preprocessing + model). |

---

## Potential Improvements

- **LSTM Modeling**: Implement Long Short-Term Memory (LSTM) networks to better capture temporal dependencies and patterns in the time-series sensor data, potentially improving failure prediction accuracy.
- **Statistical Testing**: Conduct t-tests to evaluate whether the mean values of key features differ significantly prior to failure events, offering stronger statistical validation of feature relevance. Taking into account potential feature dependencies.

