# 📖 Business Analytics System - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Structure](#data-structure)
4. [Analysis Modules](#analysis-modules)
5. [API Reference](#api-reference)
6. [Customization Guide](#customization-guide)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Business Operations & Forecasting Analytics System is an end-to-end solution for:

- **KPI Tracking**: 28+ metrics across 4 dimensions
- **Forecasting**: ML-powered 90-day predictions with 96.87% accuracy
- **Scenario Analysis**: 7 what-if business scenarios
- **Customer Analytics**: Cohort retention, LTV, churn analysis
- **Operational Insights**: Seasonality, workforce planning, regional performance

### Data Specifications

| Attribute | Value |
|-----------|-------|
| Records | 27,400 |
| Date Range | Jan 2022 - Dec 2024 |
| Categories | 5 |
| Regions | 5 |
| Segments | 4 |

---

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/business-analytics.git
cd business-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, plotly, streamlit; print('All packages installed!')"
```

---

## Data Structure

### Input Data Schema

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction date |
| year | int | Year (2022-2024) |
| month | int | Month (1-12) |
| category | string | Product category |
| region | string | Geographic region |
| segment | string | Customer segment |
| revenue | float | Revenue amount |
| cogs | float | Cost of goods sold |
| gross_profit | float | Gross profit |
| marketing_spend | float | Marketing expenses |
| orders | int | Number of orders |
| customers | int | Unique customers |

### Categories
- Electronics
- Apparel
- Home & Garden
- Food & Beverage
- Health & Beauty

### Regions
- North
- South
- East
- West
- Central

### Segments
- Enterprise (15%)
- Mid-Market (25%)
- SMB (35%)
- Consumer (25%)

---

## Analysis Modules

### 1. KPI Framework

```python
from src.COMPLETE_CODE import calculate_kpis

kpis, daily_df = calculate_kpis(df)

# Returns:
# kpis['financial'] - Revenue, margins, ROI
# kpis['customer'] - LTV, CAC, churn
# kpis['operational'] - Inventory, fill rate
# kpis['growth'] - YoY, CAGR, MoM
```

### 2. Seasonality Analysis

```python
from src.COMPLETE_CODE import analyze_seasonality

seasonality = analyze_seasonality(df)

# Returns:
# weekly_revenue - Weekly aggregated revenue
# trend - 13-week moving average
# seasonal_indices - Monthly multipliers
# dow_indices - Day-of-week patterns
```

### 3. ML Forecasting

```python
from src.COMPLETE_CODE import build_forecasting_models

forecast_data = build_forecasting_models(df)

# Models compared:
# - Linear Regression
# - Ridge Regression
# - Random Forest (best: 3.13% MAPE)
# - Gradient Boosting
```

### 4. Scenario Analysis

```python
from src.COMPLETE_CODE import scenario_analysis

scenarios = scenario_analysis(df, forecast_data)

# Scenarios:
# - Base Case
# - Marketing Boost (+30%)
# - Price Increase (+10%)
# - Cost Optimization (-15%)
# - Recession (-25% demand)
# - Expansion Strategy
# - Holiday Peak
```

---

## API Reference

### Main Functions

| Function | Parameters | Returns |
|----------|------------|---------|
| `generate_realistic_business_data()` | None | DataFrame (27,400 rows) |
| `calculate_kpis(df)` | DataFrame | dict, DataFrame |
| `analyze_seasonality(df)` | DataFrame | dict |
| `build_forecasting_models(df)` | DataFrame | dict |
| `generate_future_forecast(forecast_data, df, periods)` | dict, DataFrame, int | dict |
| `scenario_analysis(df, forecast_data)` | DataFrame, dict | DataFrame |
| `customer_churn_analysis(df)` | DataFrame | dict |

### Visualization Functions

| Function | Output |
|----------|--------|
| `plot_kpi_dashboard()` | 01_kpi_dashboard.png |
| `plot_seasonality_analysis()` | 02_seasonality_analysis.png |
| `plot_rolling_metrics()` | 03_rolling_metrics.png |
| `plot_forecasting_results()` | 04_forecasting_results.png |
| `plot_future_forecast()` | 05_future_forecast.png |
| `plot_scenario_analysis()` | 06_scenario_analysis.png |
| `plot_cost_revenue_analysis()` | 07_cost_revenue_analysis.png |
| `plot_churn_analysis()` | 08_churn_analysis.png |
| `plot_workforce_analysis()` | 09_workforce_planning.png |
| `plot_regional_category()` | 10_regional_category_analysis.png |
| `create_executive_summary()` | 11_executive_summary.png |

---

## Customization Guide

### Using Your Own Data

```python
# Replace data generation with your data
def load_custom_data(filepath):
    df = pd.read_csv(filepath)
    
    # Ensure required columns
    required = ['date', 'revenue', 'cogs', 'category', 'region', 'segment']
    for col in required:
        assert col in df.columns, f"Missing: {col}"
    
    df['date'] = pd.to_datetime(df['date'])
    return df

# In main():
df = load_custom_data('your_data.csv')
```

### Adding Custom Scenarios

```python
# In scenario_analysis() function:
scenarios = {
    'Your Scenario': {
        'marketing_change': 0.20,    # +20%
        'price_change': -0.05,       # -5%
        'cost_reduction': 0.10,      # -10%
        'demand_shock': 0.15,        # +15%
        'description': 'Q4 Strategy'
    }
}
```

### Modifying Forecast Horizon

```python
# Change from 90 to 180 days
future_forecast = generate_future_forecast(
    forecast_data, 
    df, 
    periods=180  # Now forecasts 6 months
)
```

---

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set main file: `dashboards/streamlit_app.py`
5. Deploy

### GitHub Pages

1. Settings → Pages
2. Source: main branch
3. Access: `https://username.github.io/repo/dashboards/dashboard.html`

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboards/streamlit_app.py"]
```

---

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**2. Streamlit not loading**
```bash
streamlit cache clear
streamlit run dashboards/streamlit_app.py --server.port 8502
```

**3. Charts not rendering in HTML**
- Ensure internet connection for Chart.js CDN
- Try different browser

**4. Memory issues with large data**
```python
# Reduce data size
df = df.sample(frac=0.5)  # Use 50% of data
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/business-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/business-analytics/discussions)

---

*Last updated: January 2025*
