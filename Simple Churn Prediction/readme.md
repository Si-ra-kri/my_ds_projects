# Customer Churn Prediction

This is a simple machine learning project that predicts whether a telecom customer is likely to churn based on features such as tenure, monthly charges, and total charges. The project uses logistic regression as a baseline classifier and demonstrates basic preprocessing, model training, evaluation, and interpretation.

---

## 📁 Files in This Project

- `simple_churn_prediction.py` – Main script containing data loading, preprocessing, model training, and evaluation.  
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` – Dataset used for training (not included here, downloadable from IBM sample datasets).  
- `README.md` – You're reading it!

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction
2. Run the script
   ```bash
   python simple_churn_prediction.py
## Results
Model Used: Logistic Regression

Features: tenure, MonthlyCharges, TotalCharges

Accuracy: Approximately 76%–80% (varies per run)

Evaluation Metric: Confusion Matrix is printed in console

Data Split: 80% training, 20% testing

## Key Insights
📉 Customers with longer tenure are less likely to churn (negative coefficient).

💰 Higher monthly charges are associated with higher likelihood of churn.

💳 Total charges impact churn but are correlated with tenure.

## 👤 Author
Sivaramakrishnan
GitHub – Si-ra-kri





