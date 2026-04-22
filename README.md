# ⚗️ DistillOpt — Distillation Column Optimizer

An ML-powered Streamlit dashboard for real-time prediction and optimization of a binary distillation column (benzene/toluene system).

## Features
- Live predictions using a trained Gradient Boosting model (R² = 0.990)
- Interactive sliders for all key operating parameters
- McCabe–Thiele diagram, purity contour maps, sensitivity curves
- Physically realistic synthetic dataset based on Raoult's Law + FUG method

## Project Structure
| File | Description |
|---|---|
| `app.py` | Streamlit dashboard |
| `distillation_dataset_generator.py` | Synthetic dataset generator (1000 rows) |
| `distillation_eda.py` | EDA & feature engineering pipeline |
| `distillation_model.py` | Model training, evaluation & export |
| `best_model_gradient_boosting.pkl` | Saved best model |
| `distillation_synthetic_dataset.csv` | Raw synthetic dataset |
| `distillation_processed.csv` | Processed dataset with engineered features |

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Performance
| Model | Avg R² | Avg RMSE | Avg MAE |
|---|---|---|---|
| Gradient Boosting 🏆 | 0.9902 | 538.6 | 379.3 |
| Random Forest | 0.9870 | 697.7 | 496.0 |
| Linear Regression | 0.9253 | 1795.5 | 1375.5 |

## Pipeline Overview
1. **Dataset Generation** — 1000 rows using Antoine equations, Fenske-Underwood-Gilliland, Murphree efficiency
2. **EDA & Feature Engineering** — 6 engineered features, Min-Max normalisation, 5 visualisation outputs
3. **Model Training** — Linear Regression, Random Forest, Gradient Boosting; evaluated on R², RMSE, MAE
4. **Dashboard** — Live Streamlit app with Plotly charts, sensitivity sweeps, McCabe-Thiele diagram
