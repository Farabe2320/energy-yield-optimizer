[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-yield-optimizer.streamlit.app)

# Energy-Aware Production Optimizer

A machine learning project that simultaneously optimizes **manufacturing yield** and **energy consumption** using real industrial datasets.

## Problem Statement
Manufacturing facilities face two critical challenges:
- **Yield Optimization**: Predicting and preventing product defects
- **Energy Efficiency**: Minimizing energy consumption without sacrificing output

This project solves both simultaneously using a Multi-Objective Optimization approach.

## Key Results
| Metric | Score |
|--------|-------|
| Yield Prediction F1 | 98.3% |
| Energy Forecast R² | 99.8% |
| Energy MAE | 0.677 kWh |
| Features Reduced | 562 → 100 |

## Tech Stack
- **ML Models**: XGBoost, LightGBM, Random Forest
- **Optimization**: Optuna, Scipy
- **Explainability**: SHAP
- **Dashboard**: Streamlit + Plotly
- **Data**: SECOM (UCI), Steel Industry Energy (Kaggle)

## Project Structure
```
energy-yield-optimizer/
├── data/
│   ├── raw/          
│   └── processed/    
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_optimizer.ipynb
├── app/
│   └── dashboard.py  
├── reports/
│   └── figures/      
└── requirements.txt
```

## Challenges Solved
- **Imbalanced Dataset**: 93:7 Pass/Fail ratio handled using SMOTE
- **High Dimensionality**: 562 features reduced to 100 using SelectKBest
- **Multi-Objective**: Simultaneously optimized yield and energy using Pareto analysis

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Farabe2320/energy-yield-optimizer.git
cd energy-yield-optimizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app/dashboard.py
```

## Dashboard Features
- **Dashboard Overview**: Real-time energy metrics and trends
- **Yield Predictor**: Interactive sensor parameter tuning
- **Energy Optimizer**: Find optimal operating hours and load conditions
- **Model Performance**: Compare ML model results

## Author
**Farabe2320** | Industrial & Production Engineer | ML Enthusiast  
GitHub: https://github.com/Farabe2320