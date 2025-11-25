# Retail AI Growth Engine

An end-to-end Retail Analytics app built in Python and Streamlit using the **Online Retail II** dataset.  
The app acts as a lightweight “AI Growth Copilot” for an online retailer:

- Segments customers using RFM and clustering
- Predicts short-term Customer Lifetime Value (CLV, 3-month horizon)
- Scores purchase propensity (who is likely to buy next)
- Prioritizes customers for campaigns under a budget
- Simulates price changes in a simple **Pricing Lab**
- Provides a data dictionary and visual dashboards for non-technical users

---

## 1. App Overview

The app is organized into several pages (via the Streamlit sidebar):

### 🔹 Overview
High-level business view:

- Key metrics:  
  - Total customers  
  - Average predicted CLV (3 months)  
  - Average purchase probability  
  - Number of countries in the data
- Model quality:
  - CLV model: MAE, RMSE  
  - Propensity model: Accuracy, ROC–AUC
- CLV distribution (histogram with outlier handling)
- CLV vs Purchase Probability scatter (which segments are high value & high likelihood)
- Monthly revenue trend with 3-month rolling average
- Revenue contribution by customer segment

### 🔹 Customer Explorer
Search by **Customer ID** to see:

- RFM profile: Recency, Frequency, Monetary
- Segment name (e.g. “High Value”, “At Risk”, etc.)
- Predicted CLV (next 3 months)
- Purchase probability
- Combined score (CLV × Propensity) used for targeting

### 🔹 Segments
Segment-level analytics:

- Summary table:
  - Number of customers
  - Avg Recency, Frequency, Monetary
  - Avg predicted CLV
  - Avg purchase probability
  - Total revenue & share of revenue
- Charts:
  - Segment size vs revenue
  - Average CLV by segment
  - Average propensity by segment
  - Frequency vs Monetary scatter colored by segment

### 🔹 Campaign Designer
Simple targeting engine:

- Inputs:
  - Total campaign budget
  - Cost per contact
  - Minimum purchase probability threshold
- Logic:
  - Computes **combined_score = predicted_clv × purchase_probability**
  - Filters by propensity threshold
  - Picks the top N customers under the budget
- Outputs:
  - Number of targets
  - Expected revenue (sum of predicted CLV for selected customers)
  - Expected ROI (revenue / cost)
  - Downloadable CSV with the target list

### 🔹 Wire Pricing Lab
Illustrative pricing sandbox using the transaction data:

- Select a product (by description)
- See its current average **Price**
- Slide a new price level and simulate:
  - How demand might change (based on a simple elasticity assumption)
  - Base vs new revenue
  - Delta in revenue
- Visual demand curve: **Price vs Revenue**

> ⚠️ This is a didactic pricing model, not a production-ready elasticity engine.

### 🔹 Model Diagnostics
- CLV model:
  - MAE
  - RMSE
- Propensity model:
  - Accuracy
  - ROC–AUC
- Short explanation of what each metric means for non-technical stakeholders.

### 🔹 Data Dictionary
Plain-language glossary for key fields and metrics, e.g.:

- **Recency** – days since customer’s last transaction at the reference date  
- **Frequency** – number of invoices in the feature window  
- **Monetary** – total historical revenue in the feature window  
- **Predicted CLV (3m)** – expected spend in the next 3 months  
- **Purchase Probability** – probability of at least one purchase in the next 3 months  
- **Combined Score** – CLV × probability, used to rank customers for campaigns  

---

## 2. Data

This project uses the **Online Retail II** dataset (UCI repository / Kaggle mirror), which contains transactional data for a UK-based online retail store.

Typical raw fields:

- `Invoice` / `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price` (or `UnitPrice`, depending on version)
- `Customer ID`
- `Country`

### Pre-processing

The repo includes cleaned/derived CSVs such as:

- `clean_df.csv` – cleaned transaction-level data with:
  - Positive quantity & price only
  - Added `Revenue = Quantity × Price`
- `rfm_segmented_with_names.csv` – customer-level RFM features + cluster-based segment labels
- `clv_training_data.csv` – customer-level features and **future_spend_3m** label
- `propensity_training_data.csv` – customer-level features and **purchase_next_3m** label
- `clv_model.txt`, `propensity_model.txt` – trained LightGBM models

> For another retailer, only the source transactional table needs to change; the rest of the pipeline is designed to be reusable.

---

## 3. Modeling

### CLV Model

- Task: Regression – predict **future_spend_3m** for each customer
- Input features:
  - RFM: Recency, Frequency, Monetary
  - (Optionally additional behavioral features)
- Model: **LightGBM Regressor**
- Targets:
  - Clipped high outliers
  - Sometimes log-transformed during training (`log1p`)

### Propensity Model

- Task: Binary classification – **purchase_next_3m = 1/0**
- Features: Same RFM set as CLV
- Model: **LightGBM Classifier**
- Evaluation:
  - ROC–AUC
  - Accuracy
  - Precision/Recall (viewed in notebook / diagnostics)

### Targeting Logic

For each customer:

```text
combined_score = predicted_clv * purchase_probability
