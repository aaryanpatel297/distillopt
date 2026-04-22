# ⚗️ DistillOpt — Distillation Column Optimizer

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-green?style=for-the-badge)](https://aryanpatel-distillopt.streamlit.app/)

🔗 **Live Demo:**
https://aryanpatel-distillopt.streamlit.app/

---

## Overview

DistillOpt is an interactive Streamlit-based application for analyzing and optimizing the performance of a binary distillation column (benzene–toluene system).

It combines **chemical engineering principles** with **data-driven modeling** to provide real-time insights into separation efficiency and energy consumption.

---

## Features

* Interactive control of operating parameters:

  * Feed composition (z)
  * Feed temperature (T)
  * Reflux ratio (R)
  * Column pressure (P)
  * Number of trays (N)
  * Feed flow rate (F)

* Real-time prediction of:

  * Distillate purity
  * Bottoms composition
  * Energy consumption

* Advanced visualizations:

  * McCabe–Thiele diagram
  * Purity contour maps
  * Reflux ratio sensitivity analysis
  * Tray and pressure sensitivity plots
  * Operating radar chart

* Synthetic dataset generation based on:

  * Raoult’s Law
  * Antoine equations
  * Fenske–Underwood–Gilliland (FUG) method
  * Murphree tray efficiency

---

## Live Application

The deployed app enables real-time exploration of distillation behavior by adjusting process variables and observing immediate changes in performance metrics.

👉 https://aryanpatel-distillopt.streamlit.app/

---

## Project Structure

| Path                                | Description                     |
| ----------------------------------- | ------------------------------- |
| `app.py`                            | Streamlit dashboard entry point |
| `src/dashboard/`                    | UI, plots, predictor logic      |
| `src/pipeline/dataset_generator.py` | Synthetic dataset generation    |
| `src/pipeline/eda_features.py`      | EDA & feature engineering       |
| `src/pipeline/train_model.py`       | Model training & evaluation     |
| `requirements.txt`                  | Project dependencies            |

---

## How to Run Locally

```bash
git clone https://github.com/aaryanpatel297/distillopt.git
cd distillopt
pip install -r requirements.txt
streamlit run app.py
```

---

## Model Performance

| Model                | Avg R² | Avg RMSE | Avg MAE |
| -------------------- | ------ | -------- | ------- |
| Gradient Boosting 🏆 | 0.9902 | 538.6    | 379.3   |
| Random Forest        | 0.9870 | 697.7    | 496.0   |
| Linear Regression    | 0.9253 | 1795.5   | 1375.5  |

---

## Pipeline Overview

1. **Dataset Generation**
   Synthetic dataset (1000 samples) generated using thermodynamic relations and shortcut distillation methods.

2. **EDA & Feature Engineering**
   Derived features such as relative volatility, separation factor, and energy intensity.

3. **Model Training**
   Multiple regression models trained and evaluated using R², RMSE, and MAE.

4. **Dashboard Development**
   Streamlit-based interface with Plotly visualizations for interactive analysis.

---

## Notes

* This project demonstrates the integration of **process engineering concepts** with **modern data science tools**.
* Current deployment uses a simplified predictive model for responsiveness.
* The pipeline supports extension to fully trained ML-based inference.

---

## Future Improvements

* Integrate trained ML model (`.pkl`) into live predictions
* Add real industrial dataset validation
* Expand to multicomponent distillation systems
* Include economic optimization metrics

---

## License

This project is intended for academic and educational use.
