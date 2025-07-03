# 🚲 Bike Sharing Demand Prediction

This project focuses on predicting the number of bike rentals in a given hour using historical data from the Capital Bikeshare system in Washington D.C. It explores various features such as weather conditions, seasonality, time of day, and user type. The goal is to build a regression model that can accurately forecast hourly demand.

---

## 📁 Files in This Project

- `Bike Sharing Demand Prediction.ipynb` – The main notebook containing EDA, feature engineering, model building, and evaluation.
- `train.csv` / `test.csv` – Dataset files from Kaggle (not included here due to size, but easily downloadable).
- `README.md` – Project documentation.

---

## 📦 Requirements

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
## 🛠️ How to Run the Project
1. Clone this repository:
```bash
git clone https://github.com/yourusername/bike-sharing-demand.git
cd bike-sharing-demand
```
2. Download the dataset from the Kaggle competition page and place train.csv and test.csv in the project directory.
3. Launch Jupyter Notebook:
```bash
jupyter notebook "Bike Sharing Demand Prediction.ipynb"
```
4. Run the notebook cell by cell to:

   Explore the dataset

   Perform feature engineering (e.g., extract hour, day, weekday)

   Train regression models (like Linear Regression, Random Forest, or XGBoost)

   Evaluate predictions (using RMSE or R²)

   Make final predictions on the test set

## 📊 Key Features Extracted
hour, day, month, year from datetime

workingday, holiday, weekday

weather, temperature, humidity, windspeed

Lag features or rolling means (if used)

## 👤 Author
Sivaramakrishnan
GitHub – Si-ra-kri

## ⭐ Credits
Dataset provided by Kaggle – Bike Sharing Demand

Modeling techniques inspired by standard ML workflows and Kaggle kernels





