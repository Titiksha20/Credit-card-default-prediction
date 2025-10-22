
# Credit Card Default Prediction

A production-ready machine learning pipeline for predicting credit card default risk using the UCI Credit Card Default dataset.

## Project Overview

This project implements an end-to-end ML solution for credit card default prediction, achieving **78% ROC-AUC** and **70% recall** through systematic feature engineering, model comparison, and threshold optimization.

### Key Highlights
- 30,000 customer records processed and analyzed
- 20+ engineered features capturing payment behavior and credit utilization
- 5 models trained and evaluated with robust cross-validation
- Optimized decision threshold to meet business requirements
- Production-ready Random Forest model with documented deployment strategy

## Dataset

**Source:** UCI Machine Learning Repository - Default of Credit Card Clients

**Details:**
- 30,000 records from Taiwan (2005)
- 23 initial features (demographic, payment history, bill statements)
- Binary classification: Default vs No Default
- Class imbalance: 22.12% default rate

## Project Structure
```
credit-card-default-prediction/
│
├── credit_card_default_detection.py    # Main pipeline script
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
│
├── data/
│   └── UCI_Credit_Card.csv             # Dataset (downloaded via kagglehub)
│
├── models/
│   ├── best_model.pkl                  # Trained Random Forest model
│   └── scaler.pkl                      # Fitted MinMaxScaler
│
├── visualizations/
│   ├── model_comparison_curves.png     # ROC & PR curves
│   ├── feature_importance.png          # Top features
│   └── threshold_sensitivity.png       # Threshold analysis
│
└── documentation/
    ├── project_summary.txt             # Full technical summary
    └── model_card.md                   # Model documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-default-prediction.git
cd credit-card-default-prediction

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for advanced models
pip install xgboost lightgbm

# Install kagglehub for dataset download
pip install kagglehub
```

### Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
xgboost>=1.7.0
lightgbm>=3.3.0
kagglehub>=0.1.0
```

## Usage

### Running the Full Pipeline
```bash
python credit_card_default_detection.py
```

This will:
1. Download the dataset automatically
2. Perform data cleaning and preprocessing
3. Engineer features
4. Train 5 different models
5. Optimize thresholds
6. Generate visualizations
7. Save the best model

### Quick Start Example
```python
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load and preprocess new data
new_data = pd.read_csv('new_customers.csv')
# ... apply same preprocessing steps ...

# Make predictions
probabilities = model.predict_proba(new_data_scaled)[:, 1]
predictions = (probabilities >= 0.4071).astype(int)  # Optimal threshold

print(f"Default Probability: {probabilities[0]:.2%}")
print(f"Prediction: {'Default' if predictions[0] else 'No Default'}")
```

## Methodology

### 1. Data Preprocessing
- Fixed invalid categorical codes (EDUCATION: 0,5,6 → 4; MARRIAGE: 0 → 3)
- Removed outliers using IQR method (reduced to 27,092 records)
- Smart scaling: MinMaxScaler for continuous, preserved ordinal features

### 2. Feature Engineering
Created 20+ engineered features:
- **Utilization Features:** current, average, max utilization rates
- **Payment Behavior:** max delay, chronic delayer flags, payment trends
- **Payment Ratios:** payment-to-bill ratios, underpaying indicators
- **Composite Features:** red_flag_count, high_util_delayed

### 3. Model Training
Trained 5 models with class balancing:
- Logistic Regression (baseline)
- Random Forest ⭐ (selected)
- Gradient Boosting
- XGBoost
- LightGBM

### 4. Threshold Optimization
Optimized decision thresholds to achieve:
- Minimum 70% recall (catch 70% of defaults)
- Maximize F1 score among valid thresholds

### 5. Model Selection
Selected Random Forest based on:
- Best ROC-AUC (0.7798)
- Meets recall target (70.05%)
- Stable cross-validation (0.765 ± 0.01)
- Simplicity over marginal gains

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.7798 |
| **Recall** | 0.7005 (70.05%) |
| **Precision** | 0.4079 (40.79%) |
| **F1 Score** | 0.5156 |
| **Optimal Threshold** | 0.4071 |

### Confusion Matrix (Test Set)

|  | Predicted No Default | Predicted Default |
|---|---|---|
| **Actual No Default** | 3,615 | 2,487 |
| **Actual Default** | 505 | 1,182 |

### Top 10 Important Features

1. PAY_0 (Most recent payment status)
2. PAY_2 (Payment status 2 months ago)
3. PAY_3 (Payment status 3 months ago)
4. LIMIT_BAL (Credit limit)
5. max_delay (Maximum payment delay)
6. utilization_avg (Average utilization)
7. PAY_AMT1 (Most recent payment amount)
8. PAY_4 (Payment status 4 months ago)
9. num_months_delayed (Count of delayed months)
10. AGE (Customer age)

## Visualizations

The pipeline generates several visualizations:

1. **ROC Curves** - Model comparison and performance
2. **Precision-Recall Curves** - Trade-off analysis
3. **Feature Importance** - Top predictive features
4. **Threshold Sensitivity** - Impact of threshold choice

## Business Impact

### Cost Analysis

Assuming:
- Cost of missing a default: $1,000
- Cost of false alarm: $100

**Model Performance:**
- False Negatives: 505 (missed defaults) = $505,000
- False Positives: 2,487 (false alarms) = $248,700
- **Total Cost:** $753,700
- **Cost per Customer:** $98.37

**Value Proposition:**
- Catches 70% of potential defaults
- Prevents significant financial losses
- Reduces manual review workload by prioritizing high-risk cases

## Model Interpretability

### Key Insights

1. **Payment history is king:** Recent payment status (PAY_0) is the strongest predictor
2. **Utilization matters:** High credit utilization combined with delays signals high risk
3. **Temporal patterns:** Consistent payment delays (chronic_delayer) strongly predict default
4. **Composite indicators:** red_flag_count effectively combines multiple risk signals

### Risk Indicators

High default risk when:
- Recent payment delays (PAY_0 ≥ 2)
- High utilization (>90%)
- Multiple months of delays (≥4)
- Low payment-to-bill ratios (<10%)
- Multiple red flags (≥3)



## Future Work

- [ ] Implement SHAP analysis for individual predictions
- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Add real-time monitoring dashboard
- [ ] Implement temporal validation (walk-forward)




## Acknowledgments

- Dataset: UCI Machine Learning Repository
