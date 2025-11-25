🛒 Retail AI Growth Engine
Customer Segmentation • CLV Prediction • Propensity Modeling • Pricing Lab

This project is a full end-to-end Retail Intelligence System built on the Online Retail II dataset. It integrates data cleaning, RFM segmentation, predictive modeling, pricing simulation, and a full Streamlit web app designed as if it were deployed inside a real retail business.

The goal:
✨ Turn fragmented transaction data into automated customer insights, revenue predictions, and actionable marketing decisions.

🚀 Project Highlights

✔️ 1. RFM Segmentation Engine

Creates behavioral segments using Recency, Frequency, and Monetary value.

Automatically generates human-readable segment names (e.g., “High-Value Loyalists”, “Dormant Buyers”).

✔️ 2. CLV Prediction (3-Month)

Machine Learning model using LightGBM to forecast future spending, including:

Recency

Purchase frequency

Historical spend

Log-scaled & outlier-corrected target

MAE/RMSE diagnostics


✔️ 3. Propensity-to-Purchase Model


Predicts the probability that a customer will buy again in the next 3 months.

Used to power campaign targeting and expected ROI simulations.


✔️ 4. Pricing Lab (Demand Curve Simulation)

A mini-economics lab inside the app:

Visualizes price elasticity

Forecasts revenue changes from ± price adjustments

Helps identify optimal price point


✔️ 5. Streamlit Web App


An entire UX dashboard including:

Executive Overview

Customer Explorer

Segment Analytics

Campaign Designer (CLV × Propensity)

Pricing Lab

Model Diagnostics

Data Dictionary

This mirrors the real structure of a Growth & CRM Analytics tool used in modern retail.

🏗 Tech Stack

Layer	Tools

Data Processing	Pandas, NumPy

Modeling	LightGBM, Scikit-learn

Visualization	Plotly, Streamlit Charts

Web App	Streamlit

Version Control	GitHub

Deployment	Streamlit Cloud

📂 Project Structure
/app.py                      → Main Streamlit app  
/clean_df.csv                → Cleaned transaction data  
/rfm_segmented_with_names.csv → Customer segmentation table  
/clv_training_data.csv        → Model training dataset  
/propensity_training_data.csv → Propensity model dataset  
/clv_model.txt                → Trained LightGBM CLV model  
/propensity_model.txt         → Trained LightGBM propensity model  


🏬 How This Would Work in a Real Retail Business

This project reflects how retailers operate in real life — not as a one-off analysis, but as a living decision engine.

1. Daily/Weekly Data Pipeline

In a real business, new transaction data from POS or e-commerce feeds would refresh automatically:

New invoices

Product prices

Returns

Customer IDs & loyalty data

A scheduled job (Airflow/Databricks/Cloud Functions) would update:

RFM scores

CLV predictions

Propensity scores


2. CRM & Marketing Teams Use the Dashboard

Teams can:

Identify high-value customers who are slipping

Build retention campaigns

Compare segments by revenue contribution

Download AI-optimized targeting lists

    The “Campaign Designer” mirrors how:

        Sephora

        Starbucks
        
        Amazon

        Walmart
optimize CRM campaigns inside their internal systems.

3. Pricing Team Uses the Pricing Lab

Pricing teams simulate revenue changes from:

10% discount

Repricing bundles

Peak-season markup

Price elasticity testing

This mirrors tools used in:

Retail merchandising

Promotions planning

Dynamic pricing teams

4. Leadership Uses Executive View

Executives can monitor:

Average CLV

Segment health

Revenue trends

Churn risk


This becomes a single source of truth for customer strategy.

📈 Business Impact (If Deployed)
Area	Impact

Marketing	15–35% improvement in campaign ROI

CRM	Higher retention, better personalization

Finance	Forecastable revenue via CLV

Pricing	Evidence-based decisions vs. guesswork

Leadership	Clear visibility into customer behavior


💼 Real-World Applications

This system could be used by:

Fashion & apparel brands

E-commerce stores

Consumer electronics retailers

Specialty retail (cosmetics, gifts, FMCG)

Subscription commerce


