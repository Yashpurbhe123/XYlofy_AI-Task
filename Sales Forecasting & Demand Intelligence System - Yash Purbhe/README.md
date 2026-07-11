# 🔮 Sales Forecasting & Demand Intelligence System

Welcome to the **Sales Forecasting & Demand Intelligence System** project. This is a multi-layered, production-grade supply chain intelligence system built to analyze retail transaction data, predict future product demand, detect anomalous sales patterns, and profile inventory categories using machine learning.

The system is designed to solve the critical retail challenge: **optimizing inventory positioning** to prevent expensive stockouts while minimizing tied-up capital and warehousing overhead.

---

## 🚀 Key Features

### 1. Data Intelligence & Exploration (Task 1 & 2)
- Parses transaction temporal fields (Year, Month, Week, Day of Week, Quarter, Season).
- Generates seasonal decompositions, breaking sales signals down into **Trend, Seasonality, and Residual Noise** using `statsmodels`.
- Tests for stationarity using the **Augmented Dickey-Fuller (ADF) Test** to support time-series modeling.

### 2. Multi-Model Forecasting Engine (Task 3 & 4)
Builds and compares three fundamentally different forecasting methods on monthly sales:
1. **SARIMA (Seasonal ARIMA):** Classical statistical approach capturing seasonal cycles.
2. **Facebook Prophet:** Additive regression model focusing on trend growth and holiday impacts.
3. **XGBoost Regressor:** Supervised machine learning approach utilizing lag features (Lag 1, Lag 2, Lag 3) and rolling averages.
- Recommends the best-performing model based on **MAE, RMSE, and MAPE** error metrics.
- Extends the best model to generate 3-month forecast projections for individual regions and product categories.

### 3. Anomaly Detection & RCA (Task 5)
- Uses **Isolation Forest** (unsupervised ML) and **Rolling Z-Score** (statistical) to flag weeks with highly irregular sales spikes or dips.
- Runs an automated **Root Cause Analysis (RCA) Engine** that drills down into transaction records during anomaly windows to extract the primary category, state, and largest customer transactions driving the variance.

### 4. Demand Segmentation & Inventory Profiling (Task 6)
- Clusters product sub-categories using **K-Means Clustering** based on total volume, average order values, demand growth, and volatility.
- Profiles clusters into actionable inventory categories:
  - **🏆 Champions:** High Volume & Stable Demand $\to$ continuous replenishment, low safety stock.
  - **🌪️ Volatile Hits:** Seasonal Demand $\to$ dynamic buffer adjustment.
  - **💎 High-Value Niche:** Premium Items $\to$ Just-in-Time vendor sourcing.
  - **🐢 Slow Movers:** Long Tail $\to$ catalog pruning and liquidation.

### 5. Interactive Streamlit Dashboard (Task 7)
- **Aesthetic Theme:** Dark-slate glassmorphic theme, utilizing Google Font `Outfit`, custom gradient HTML metric tiles, and styled Plotly charts.
- **Pages:**
  - **Overview:** Interactive metrics, USA State choropleth map, category sunburst, and monthly area trend.
  - **Forecast Explorer:** Interactive segment selector, horizon sliders (1-12 months), and model selector (SARIMA vs Holt-Winters) with backtest metrics and CSV export.
  - **Anomaly Report:** Adjustable contamination sensitivity, aggregation settings (weekly/monthly), and automated RCA logs.
  - **Product Segments:** Cluster count sliders, 2D PCA plots, and strategic stocking cards.

---

## 🛠️ Tech Stack & Dependencies
- **Core:** Python 3.10+
- **Data Engineering:** `pandas`, `numpy`
- **Predictive Modeling:** `statsmodels` (SARIMA, Exponential Smoothing), `prophet`, `xgboost`
- **Unsupervised ML:** `scikit-learn` (KMeans, IsolationForest, StandardScaler, PCA)
- **Visualizations:** `plotly`, `matplotlib`, `seaborn`
- **Dashboard Deployment:** `streamlit`

---

## 📂 Project Structure

```
Sales Forecasting & Demand Intelligence System/
│
├── Analysis.ipynb         # Comprehensive Jupyter Notebook (Tasks 1-6)
├── Train.csv              # Primary Superstore sales dataset
├── app.py                 # Streamlit interactive dashboard code (Task 7)
├── requirements.txt       # Python libraries and versions list
│
├── charts/                # Generated PNG charts from notebook analysis
│   ├── 1-Total Revenue by Category.png
│   ├── 10- Prophet Forecast.png
│   ├── 16- Anomaly Detection.png
│   └── ... (18 files total)
│
└── README.md              # Project documentation (this file)
```

---

## ⚙️ How to Run

### 1. Installation
Clone this repository or navigate to the project directory, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Analysis
To explore the core data science tasks (Tasks 1 to 6), run Jupyter Notebook or JupyterLab:
```bash
jupyter notebook Analysis.ipynb
```

### 3. Start the Interactive Dashboard
To launch the Streamlit dashboard locally, run the following command in your terminal:
```bash
streamlit run app.py
```
This will spin up a local development server and automatically open the application in your default web browser (typically at `http://localhost:8501`).

---

## 📊 Summary of Model Performance (Task 3)
During evaluation, the models generated the following aggregate performance metrics:

| Model | MAE | RMSE | MAPE (%) | Month 1 Forecast | Month 2 Forecast | Month 3 Forecast |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **SARIMA (Recommended)** | **$19,244.49** | **$19,950.07** | **20.53%** | **$60,331.79** | **$91,458.22** | **$97,167.57** |
| **Prophet** | $20,296.01 | $22,487.47 | 21.89% | $51,083.66 | $90,045.40 | $89,661.19 |
| **XGBoost** | $22,361.16 | $30,400.84 | 21.07% | $86,172.48 | $66,469.30 | $76,140.10 |

*Based on achieving the lowest overall Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE), **SARIMA** is recommended for production use.*
