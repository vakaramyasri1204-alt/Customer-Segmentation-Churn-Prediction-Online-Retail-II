# Customer Segmentation & Churn Prediction — Online Retail II

## Overview
This project analyzes ~1M+ transactions from a UK-based online retailer (Dec 2009–Dec 2011) to segment customers by purchasing behavior and predict which customers are at risk of churning. The goal is to give the business an early-warning system for customer attrition, so retention efforts can be targeted rather than spread evenly across the customer base.

**Dataset:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — UCI Machine Learning Repository
**Tools:** Python, Pandas, Scikit-learn, Matplotlib/Seaborn

## Business Problem
Customer acquisition is expensive; retaining existing customers is cheaper and more profitable. The business needs a way to:
1. Identify its most valuable customers (to protect and reward them)
2. Predict which customers are likely to stop purchasing (to intervene before they leave)
3. Understand *why* customers churn, to fix the underlying drivers

## Methodology

### 1. Data Cleaning
- Removed transactions with missing Customer ID (can't attribute behavior without a known customer)
- Removed cancelled orders (Invoice numbers starting with "C") and negative/zero quantities
- Calculated revenue per line item (Quantity × Price)

### 2. RFM Segmentation
Scored each customer 1–5 on three dimensions using quintiles:
- **Recency** — days since last purchase
- **Frequency** — number of distinct orders
- **Monetary** — total amount spent

Combined scores were used to bucket customers into segments: Champions, Loyal Customers, At Risk, Lost/Low Value.

### 3. Churn Definition
A customer is defined as **churned** if they have not purchased in 90+ days relative to the snapshot date. This threshold is a modeling assumption — in a real engagement it would be validated against the client's actual average repurchase cycle.

### 4. Feature Engineering
In addition to RFM, added: total items purchased, average order value, distinct products purchased, and customer tenure (days between first and last purchase).

### 5. Modeling
Trained and compared two classifiers:
- **Logistic Regression** — interpretable baseline, coefficients are business-explainable
- **Random Forest** — captures non-linear relationships, generally higher accuracy

Evaluated using precision/recall/F1 (classification report) and ROC-AUC.

## Key Findings
*(fill in with your actual results once you have them)*
- Logistic Regression ROC-AUC: `___`
- Random Forest ROC-AUC: `___`
- Top churn drivers (by feature importance): `___`
- % of customers classified as high-risk: `___`

## Business Recommendations
*(example structure — replace with your actual findings)*
- **Champions** (top RFM segment): protect with a loyalty/VIP program; they drive a disproportionate share of revenue
- **At Risk**: target with win-back campaigns (e.g. targeted discount or re-engagement email) before they fully churn
- **New/low-tenure customers with few orders**: highest churn risk per the model — prioritize early-engagement touchpoints in the first 90 days
- **Lost/Low Value**: deprioritize marketing spend; not cost-effective to win back

## Limitations
- No external data (marketing touchpoints, customer service history, competitor activity) — model relies only on transaction data
- Churn threshold (90 days) is a judgment call, not validated against real repurchase cycles
- Class imbalance (churners typically a minority) can inflate accuracy while masking weak recall — worth checking/addressing with techniques like class weighting or resampling
- Analysis covers a single ~2-year window; seasonality effects (e.g. holiday season) may bias results

## Next Steps
- Validate churn threshold against actual repurchase cycle data
- Test class-balancing techniques (SMOTE, class weights) to improve recall on the minority (churned) class
- Extend with a time-series demand forecast to complement the churn model
- Feed segments into the SQL/Power BI portions of this portfolio for a full end-to-end view

## Files
- `notebook.ipynb` — full analysis (cleaning, RFM, churn model)
- `README.md` — this summary

