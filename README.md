# Fraud Detection System using XGBoost

This project implements a binary classification model to detect fraudulent trades based on user account behavior and payment metadata.

## 📊 Dataset Overview
The model was trained on a dataset of ~39,000 transactions. 
* **Legitimate Trades (Class 0):** 38,661
* **Fraudulent Trades (Class 1):** 560
* **Imbalance Ratio:** ~69:1



## 🛠️ Technical Implementation
### Features Used:
* `accountAgeDays`: Age of the user account.
* `numItems`: Number of items in the transaction.
* `localTime`: The hour/time of the transaction.
* `paymentMethod`: Categorical data (Credit Card, PayPal, Store Credit) handled via One-Hot Encoding.
* `paymentMethodAgeDays`: How long the payment method has been linked to the account.

### Model:
I utilized the **XGBoost Classifier** due to its efficiency with tabular data and its built-in support for imbalanced datasets.




## 🖥️ How to Run
1. Clone the repository.
2. Ensure you have a `trades.csv` in the root directory (or use a dummy generator).
3. Install dependencies:
   ```bash
   pip install pandas xgboost scikit-learn
