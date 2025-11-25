import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# Try to import sklearn safely
try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import lightgbm safely
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    lgb = None
    LGB_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail AI Growth Engine",
    layout="wide",
)

# ---------------------------------------------------------
# DATA / MODEL LOADERS
# ---------------------------------------------------------

def load_clean_transactions():
    return pd.read_csv("clean_df.csv", parse_dates=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def load_rfm_segments():
    return pd.read_csv("rfm_segmented_with_names.csv")

def load_clv_training():
    return pd.read_csv("clv_training_data.csv")

def load_propensity_training():
    return pd.read_csv("propensity_training_data.csv")

def load_clv_model():
    if not LGB_AVAILABLE:
        raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
    return lgb.Booster(model_file="clv_model.txt")

def load_propensity_model():
    if not LGB_AVAILABLE:
        raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
    return lgb.Booster(model_file="propensity_model.txt")


# ---------------------------------------------------------
# MASTER CUSTOMER TABLE
# ---------------------------------------------------------

def build_master_customer_table():
    rfm = load_rfm_segments()
    clv_train = load_clv_training()
    prop_train = load_propensity_training()

    clv_train = clv_train[["Customer ID", "Recency", "Frequency", "Monetary", "future_spend_3m"]]
    prop_train = prop_train[["Customer ID", "purchase_next_3m"]]

    df = rfm.merge(
        clv_train,
        on=["Customer ID", "Recency", "Frequency", "Monetary"],
        how="left"
    )
    df = df.merge(prop_train, on="Customer ID", how="left")

    X = df[["Recency", "Frequency", "Monetary"]]

    clv_model = load_clv_model()
    df["predicted_clv"] = np.expm1(clv_model.predict(X))

    prop_model = load_propensity_model()
    df["purchase_probability"] = prop_model.predict(X)

    df["combined_score"] = df["predicted_clv"] * df["purchase_probability"]

    return df


# ---------------------------------------------------------
# PRICING / ELASTICITY HELPERS
# ---------------------------------------------------------

def build_pricing_table(clean_tx, min_weeks=8, top_n=30):
    """
    Build a table with basic price elasticity estimates
    for top-N products by revenue.
    """
    df = clean_tx.copy()

    # ensure we only use positive prices / quantities
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # week bucket
    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time

    # top products by total revenue
    prod_revenue = (
        df.groupby("StockCode")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    df = df[df["StockCode"].isin(prod_revenue)]

    rows = []

    for code, g in df.groupby("StockCode"):
        # weekly aggregates for this product
        weekly = g.groupby("Week").agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        )
        weekly["AvgPrice"] = weekly["Revenue"] / weekly["Quantity"]

        # filter out bad weeks
        weekly = weekly[(weekly["Quantity"] > 0) & (weekly["AvgPrice"] > 0)]

        if len(weekly) < min_weeks:
            continue
        if weekly["AvgPrice"].std() < 1e-6:
            # basically no price variation -> can't estimate elasticity
            continue

        # log-log regression: log(Q) = a + b log(P)
        x = np.log(weekly["AvgPrice"].values)
        y = np.log(weekly["Quantity"].values)

        # simple linear regression using numpy
        b, a = np.polyfit(x, y, 1)  # slope, intercept
        elasticity = float(b)       # usually negative

        # summary stats
        avg_price = weekly["AvgPrice"].mean()
        avg_qty = weekly["Quantity"].mean()
        total_rev = weekly["Revenue"].sum()
        weeks = len(weekly)

        # description: most common description for this product
        desc = g["Description"].mode().iloc[0] if not g["Description"].mode().empty else ""

        # simple sensitivity tag
        if elasticity >= -0.5:
            sens = "Low sensitivity"
        elif elasticity >= -1.5:
            sens = "Medium sensitivity"
        else:
            sens = "High sensitivity"

        rows.append({
            "StockCode": code,
            "Description": desc,
            "Weeks": weeks,
            "AvgPrice": avg_price,
            "AvgQty": avg_qty,
            "TotalRevenue": total_rev,
            "Elasticity": elasticity,
            "Sensitivity": sens
        })

    if not rows:
        return pd.DataFrame()

    pricing_df = pd.DataFrame(rows).sort_values("TotalRevenue", ascending=False)
    return pricing_df


def get_product_weekly_series(clean_tx, stock_code):
    """Weekly price & quantity series for a single product (for charts)."""
    df = clean_tx.copy()
    df = df[(df["StockCode"] == stock_code) &
            (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    if df.empty:
        return pd.DataFrame()

    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    weekly = df.groupby("Week").agg(
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum")
    )
    weekly["AvgPrice"] = weekly["Revenue"] / weekly["Quantity"]
    weekly = weekly[(weekly["Quantity"] > 0) & (weekly["AvgPrice"] > 0)]
    weekly = weekly.sort_index()
    return weekly

# ---------------------------------------------------------
# METRICS (NO squared= ANYWHERE)
# ---------------------------------------------------------

def compute_clv_metrics():
    clv = load_clv_training()
    model = load_clv_model()

    X = clv[["Recency", "Frequency", "Monetary"]]
    y_true = clv["future_spend_3m"].values
    y_pred = np.expm1(model.predict(X))

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse


def compute_propensity_metrics():
    prop = load_propensity_training()
    model = load_propensity_model()

    X = prop[["Recency", "Frequency", "Monetary"]]
    y_true = prop["purchase_next_3m"].values
    proba = model.predict(X)
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, proba)
    return acc, auc


# ---------------------------------------------------------
# PAGES
# ---------------------------------------------------------

def page_overview(master_df, clean_tx):
    st.title("Retail AI Growth Engine – Overview")
    st.caption(
        "Customer segmentation, CLV prediction, and campaign targeting for the Online Retail II dataset."
    )

    # ===== TOP METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{master_df.shape[0]:,}")
    col2.metric("Avg Predicted CLV (3m)", f"{master_df['predicted_clv'].mean():.2f}")
    col3.metric("Avg Purchase Probability", f"{master_df['purchase_probability'].mean():.2f}")
    col4.metric("Countries in data", clean_tx["Country"].nunique())

    # ===== MODEL QUALITY =====
    st.subheader("Model Quality")

    mae, rmse = compute_clv_metrics()
    acc, auc = compute_propensity_metrics()

    c1, c2 = st.columns(2)
    with c1:
        st.write("**CLV Model (Regression)**")
        st.write(f"- MAE: `{mae:.2f}`")
        st.write(f"- RMSE: `{rmse:.2f}`")
    with c2:
        st.write("**Propensity Model (Classification)**")
        st.write(f"- Accuracy: `{acc:.3f}`")
        st.write(f"- ROC–AUC: `{auc:.3f}`")

    # ===== CLV DISTRIBUTION (HISTOGRAM) =====
    st.subheader("CLV Distribution (3-Month Prediction)")

    clv = master_df["predicted_clv"].clip(
        upper=master_df["predicted_clv"].quantile(0.99)
    )
    clv_df = pd.DataFrame({"CLV_3m": clv})

    clv_chart = (
        alt.Chart(clv_df)
        .mark_bar(opacity=0.8)
        .encode(
            alt.X("CLV_3m:Q", bin=alt.Bin(maxbins=30), title="Predicted CLV (3 months)"),
            alt.Y("count()", title="Number of Customers"),
        )
        .properties(height=300)
    )

    st.altair_chart(clv_chart, use_container_width=True)

    st.caption(
        f"Median CLV: {clv.median():.2f} • "
        f"Average CLV: {clv.mean():.2f} • "
        f"Top 10% customers contribute ≈ "
        f"{master_df.nlargest(int(0.1*len(master_df)), 'predicted_clv')['predicted_clv'].sum() / clv.sum():.0%} of predicted revenue."
    )

    # ===== CLV vs. PROPENSITY SCATTER =====
    st.subheader("CLV vs Purchase Probability")

    scatter_df = master_df[["predicted_clv", "purchase_probability", "SegmentName"]].copy()
    scatter_df = scatter_df.sample(min(3000, len(scatter_df)), random_state=42)

    scatter_chart = (
        alt.Chart(scatter_df)
        .mark_circle(opacity=0.6, size=40)
        .encode(
            x=alt.X("predicted_clv:Q", title="Predicted CLV (3m)"),
            y=alt.Y("purchase_probability:Q", title="Purchase Probability (next 3m)"),
            color=alt.Color("SegmentName:N", title="Segment"),
            tooltip=[
                alt.Tooltip("predicted_clv:Q", format=".2f", title="CLV (3m)"),
                alt.Tooltip("purchase_probability:Q", format=".2f", title="Probability"),
                alt.Tooltip("SegmentName:N", title="Segment"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(scatter_chart, use_container_width=True)
    st.caption(
        "Each point is a customer. Top-right corner = high value and high likelihood to buy "
        "— ideal campaign targets."
    )

    # ===== REVENUE OVER TIME (MONTHLY + ROLLING MEAN) =====
    st.subheader("Revenue Over Time (Monthly)")

    tx = clean_tx.copy()
    tx = tx.set_index("InvoiceDate").sort_index()
    if "Revenue" not in tx.columns:
        tx["Revenue"] = tx["Quantity"] * tx["UnitPrice"]

    monthly = tx["Revenue"].resample("M").sum().rename("Revenue")
    monthly_df = monthly.reset_index()
    monthly_df["Rolling3M"] = monthly_df["Revenue"].rolling(window=3).mean()

    rev_chart = (
        alt.Chart(monthly_df)
        .transform_fold(
            ["Revenue", "Rolling3M"], as_=["Metric", "Value"]
        )
        .mark_line()
        .encode(
            x=alt.X("InvoiceDate:T", title="Month"),
            y=alt.Y("Value:Q", title="Revenue"),
            color=alt.Color(
                "Metric:N",
                title="Metric",
                scale=alt.Scale(range=["#4C72B0", "#55A868"]),
            ),
            tooltip=[
                alt.Tooltip("InvoiceDate:T", title="Month"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Value:Q", format=".2f", title="Value"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(rev_chart, use_container_width=True)
    st.caption(
        "Blue line shows monthly revenue; green line is a 3-month moving average to highlight trend."
    )

    # ===== REVENUE BY SEGMENT =====
    st.subheader("Revenue by Segment")

    seg_rev = (
        master_df.groupby("SegmentName")["Monetary"]
        .sum()
        .sort_values(ascending=False)
        .rename("Revenue")
        .reset_index()
    )
    seg_rev["Share"] = seg_rev["Revenue"] / seg_rev["Revenue"].sum()

    seg_chart = (
        alt.Chart(seg_rev)
        .mark_bar()
        .encode(
            x=alt.X("Revenue:Q", title="Revenue"),
            y=alt.Y("SegmentName:N", sort="-x", title="Segment"),
            color=alt.Color("SegmentName:N", legend=None),
            tooltip=[
                alt.Tooltip("SegmentName:N", title="Segment"),
                alt.Tooltip("Revenue:Q", format=".2f", title="Revenue"),
                alt.Tooltip("Share:Q", format=".0%", title="Share of total"),
            ],
        )
        .properties(height=40 * len(seg_rev))
    )

    st.altair_chart(seg_chart, use_container_width=True)
    st.caption("Segments are ordered by total revenue contribution.")


def page_customer_explorer(master_df):
    st.title("Customer Explorer")
    st.caption("Inspect RFM profile, CLV and purchase probability for a single customer.")

    ids = master_df["Customer ID"].dropna().sort_values().astype(int).tolist()
    if not ids:
        st.warning("No customers found.")
        return

    selected_id = st.selectbox("Select Customer ID", ids)
    row = master_df[master_df["Customer ID"] == selected_id]

    if row.empty:
        st.error("Customer not found.")
        return

    row = row.iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Recency (days)", int(row["Recency"]))
        st.metric("Segment", row["SegmentName"])
    with c2:
        st.metric("Frequency (# invoices)", int(row["Frequency"]))
        st.metric("Monetary (past spend)", f"{row['Monetary']:.2f}")
    with c3:
        st.metric("Predicted CLV (3m)", f"{row['predicted_clv']:.2f}")
        st.metric("Purchase Probability", f"{row['purchase_probability']:.2f}")

    st.subheader("Combined Score (Targeting Priority)")
    st.write(
        f"For this customer, the combined score `CLV × probability` is **{row['combined_score']:.2f}**. "
        "Higher scores indicate better candidates for campaigns."
    )

    st.subheader("Raw Feature Row")
    st.dataframe(row.to_frame().T, use_container_width=True)


def page_segments(master_df):
    st.title("Customer Segments")

    st.caption(
        "Compare how different segments behave in terms of size, value, and propensity to buy."
    )

    # ===== SEGMENT SUMMARY TABLE =====
    seg_summary = (
        master_df.groupby("SegmentName")
        .agg(
            Customers=("Customer ID", "nunique"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgPredCLV=("predicted_clv", "mean"),
            AvgPropensity=("purchase_probability", "mean"),
            TotalRevenue=("Monetary", "sum"),
        )
        .sort_values("TotalRevenue", ascending=False)
    )

    seg_summary["RevenueShare"] = (
        seg_summary["TotalRevenue"] / seg_summary["TotalRevenue"].sum()
    )

    st.dataframe(
        seg_summary.style.format(
            {
                "AvgRecency": "{:.1f}",
                "AvgFrequency": "{:.2f}",
                "AvgMonetary": "{:.2f}",
                "AvgPredCLV": "{:.2f}",
                "AvgPropensity": "{:.2f}",
                "TotalRevenue": "{:.2f}",
                "RevenueShare": "{:.0%}",
            }
        )
    )

    # ===== SEGMENT SIZE vs REVENUE =====
    st.subheader("Segment Size vs Revenue")

    seg_plot_df = seg_summary.reset_index()

    size_rev_chart = (
        alt.Chart(seg_plot_df)
        .transform_fold(
            ["Customers", "TotalRevenue"], as_=["Metric", "Value"]
        )
        .mark_bar()
        .encode(
            x=alt.X("Value:Q", title="Value"),
            y=alt.Y("SegmentName:N", sort="-x", title="Segment"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("SegmentName:N", title="Segment"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Value:Q", format=".2f", title="Value"),
            ],
        )
        .properties(height=40 * len(seg_plot_df))
    )

    st.altair_chart(size_rev_chart, use_container_width=True)
    st.caption(
        "Some segments may be small but highly lucrative; others are large but low value."
    )

    # ===== CLV BY SEGMENT =====
    st.subheader("Average CLV by Segment")

    clv_chart = (
        alt.Chart(seg_plot_df)
        .mark_bar()
        .encode(
            x=alt.X("AvgPredCLV:Q", title="Average Predicted CLV (3m)"),
            y=alt.Y("SegmentName:N", sort="-x", title="Segment"),
            color=alt.Color("SegmentName:N", legend=None),
            tooltip=[
                alt.Tooltip("SegmentName:N", title="Segment"),
                alt.Tooltip("AvgPredCLV:Q", format=".2f", title="Avg CLV"),
            ],
        )
        .properties(height=40 * len(seg_plot_df))
    )

    st.altair_chart(clv_chart, use_container_width=True)

    # ===== PROPENSITY BY SEGMENT =====
    st.subheader("Average Purchase Probability by Segment")

    prop_chart = (
        alt.Chart(seg_plot_df)
        .mark_bar()
        .encode(
            x=alt.X("AvgPropensity:Q", title="Average purchase probability"),
            y=alt.Y("SegmentName:N", sort="-x", title="Segment"),
            color=alt.Color("SegmentName:N", legend=None),
            tooltip=[
                alt.Tooltip("SegmentName:N", title="Segment"),
                alt.Tooltip("AvgPropensity:Q", format=".2f", title="Avg probability"),
            ],
        )
        .properties(height=40 * len(seg_plot_df))
    )

    st.altair_chart(prop_chart, use_container_width=True)

    # ===== SCATTER: FREQUENCY vs MONETARY =====
    st.subheader("Customer Scatter: Frequency vs Monetary by Segment")

    scatter_df = master_df[["Frequency", "Monetary", "SegmentName"]].copy()
    scatter_df = scatter_df.sample(min(3000, len(scatter_df)), random_state=42)

    scatter_chart = (
        alt.Chart(scatter_df)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("Frequency:Q", title="Frequency (# invoices)"),
            y=alt.Y("Monetary:Q", title="Monetary (historical spend)"),
            color=alt.Color("SegmentName:N", title="Segment"),
            tooltip=[
                alt.Tooltip("Frequency:Q", title="Frequency"),
                alt.Tooltip("Monetary:Q", format=".2f", title="Monetary"),
                alt.Tooltip("SegmentName:N", title="Segment"),
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(scatter_chart, use_container_width=True)
    st.caption(
        "High-frequency, high-spend customers cluster in the top-right; these are your core loyalty segments."
    )


def page_campaign_designer(master_df):
    st.title("Campaign Designer")
    st.caption("Use CLV × propensity to choose which customers to target under a budget.")

    with st.sidebar:
        st.subheader("Campaign Settings")
        budget = st.number_input("Total budget ($)", value=5000.0, min_value=0.0, step=500.0)
        cost_per_contact = st.number_input("Cost per contact ($)", value=2.0, min_value=0.0, step=0.5)
        min_prob = st.slider("Minimum purchase probability", 0.0, 1.0, 0.3, 0.05)

    candidates = master_df[master_df["purchase_probability"] >= min_prob].copy()
    if candidates.empty:
        st.warning("No customers meet the minimum probability threshold.")
        return

    max_customers = int(budget // cost_per_contact) if cost_per_contact > 0 else candidates.shape[0]
    max_customers = min(max_customers, candidates.shape[0])

    targets = candidates.sort_values("combined_score", ascending=False).head(max_customers)

    expected_revenue = targets["predicted_clv"].sum()
    expected_cost = max_customers * cost_per_contact
    roi = expected_revenue / expected_cost if expected_cost > 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Targets Selected", max_customers)
    c2.metric("Expected Revenue", f"{expected_revenue:.2f}")
    c3.metric("Expected ROI (Revenue / Cost)", f"{roi:.2f}")

    st.subheader("Target Customer List")
    st.dataframe(
        targets[
            ["Customer ID", "SegmentName", "Recency", "Frequency",
             "Monetary", "predicted_clv", "purchase_probability", "combined_score"]
        ].reset_index(drop=True),
        use_container_width=True,
    )

    csv = targets.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download target list (CSV)",
        data=csv,
        file_name="campaign_target_list.csv",
        mime="text/csv",
    )


def page_model_diagnostics():
    st.title("Model Diagnostics")
    st.caption("Quick health check for CLV and propensity models.")

    mae, rmse = compute_clv_metrics()
    acc, auc = compute_propensity_metrics()

    st.subheader("CLV Model (Regression)")
    st.write(f"- **MAE**: `{mae:.2f}` – average absolute error on 3-month spend.")
    st.write(f"- **RMSE**: `{rmse:.2f}` – penalizes large errors more than MAE.")

    st.subheader("Propensity Model (Classification)")
    st.write(f"- **Accuracy**: `{acc:.3f}` – share of customers correctly classified.")
    st.write(f"- **ROC–AUC**: `{auc:.3f}` – ranking quality of likely buyers vs non-buyers.")

# ---------------------------------------------------------
# PAGE: PRICING LAB
# ---------------------------------------------------------

def page_pricing_lab(clean_tx, pricing_df):
    st.title("Pricing Lab")
    st.caption("Explore basic price sensitivity for top products and run simple what-if scenarios.")

    if pricing_df.empty:
        st.warning("Not enough price variation to estimate elasticity. "
                   "Try increasing your time window or checking the data.")
        return

    # product selector
    options = (
        pricing_df["StockCode"].astype(str) +
        " – " +
        pricing_df["Description"].str.slice(0, 40)
    )
    mapping = dict(zip(options, pricing_df["StockCode"]))

    selected_label = st.selectbox("Choose a product", options)
    stock_code = mapping[selected_label]

    prod_row = pricing_df[pricing_df["StockCode"] == stock_code].iloc[0]

    st.subheader("Product Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("StockCode", stock_code)
    c1.metric("Weeks observed", int(prod_row["Weeks"]))
    c2.metric("Avg Price", f"{prod_row['AvgPrice']:.2f}")
    c2.metric("Avg Weekly Qty", f"{prod_row['AvgQty']:.1f}")
    c3.metric("Total Revenue", f"{prod_row['TotalRevenue']:.2f}")
    c3.metric("Elasticity (approx.)", f"{prod_row['Elasticity']:.2f}")

    st.info(
        f"Sensitivity tag: **{prod_row['Sensitivity']}**  \n"
        "Interpretation: elasticity is from a simple log–log regression on historical prices and quantities. "
        "This is an approximation, not a causal estimate."
    )

    # weekly series charts
    weekly = get_product_weekly_series(clean_tx, stock_code)
    if weekly.empty:
        st.warning("No weekly data for this product.")
        return

    st.subheader("Historical Price and Quantity")

    c4, c5 = st.columns(2)
    with c4:
        st.line_chart(weekly["AvgPrice"], y_label="Avg Price", x_label="Week")
    with c5:
        st.line_chart(weekly["Quantity"], y_label="Units Sold", x_label="Week")

    # --- what-if simulator ---
    st.subheader("What-if Price Scenario")

    elastic = prod_row["Elasticity"]
    avg_price = prod_row["AvgPrice"]
    avg_qty = prod_row["AvgQty"]
    current_revenue = avg_price * avg_qty

    delta = st.slider(
        "Price change (%)",
        min_value=-30,
        max_value=30,
        value=0,
        step=5
    )

    pct = delta / 100.0
    new_price = avg_price * (1 + pct)

    # Q_new ≈ Q0 * ( (1 + ΔP) ** elasticity )
    # ensure we don't blow up on weird floats
    new_qty = avg_qty * (1 + pct) ** elastic if (1 + pct) > 0 else 0.0
    new_revenue = new_price * new_qty

    c6, c7, c8 = st.columns(3)
    c6.metric("Current Revenue (per week)", f"{current_revenue:.2f}")
    c7.metric("New Revenue (per week)", f"{new_revenue:.2f}")
    change = (new_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0.0
    c8.metric("Revenue change (%)", f"{change:.1f}%")

    st.caption(
        "This uses elasticity × baseline price/volume to approximate how revenue could change "
        "for small price movements. In production, you would validate this with experiments."
    )

def page_wire_pricing_lab(clean_tx):
    st.title("Wire Pricing Lab – Pricing Simulation")

    st.write("""
    This tool helps explore how different pricing strategies could affect revenue.
    Select a product and simulate price changes.
    """)

    # Make sure column names match your dataset
    if "Description" not in clean_tx.columns or "Price" not in clean_tx.columns:
        st.error("Your dataset must include: Description, Price")
        st.stop()

    # Product selection
    product_list = clean_tx["Description"].dropna().unique()
    product_name = st.selectbox("Select a product", sorted(product_list))

    # Filter the product’s data
    product_df = clean_tx[clean_tx["Description"] == product_name]

    if product_df.empty:
        st.warning("No data available for this product.")
        return

    # Base price (using Price column)
    base_price = product_df["Price"].mean()

    st.subheader("Base Price")
    st.metric("Current Avg Price", f"${base_price:.2f}")

    # User simulation
    st.subheader("Simulate Price Change")
    new_price = st.slider(
        "Select a new price:",
        min_value=float(base_price * 0.5),
        max_value=float(base_price * 2.0),
        value=float(base_price),
        step=0.10
    )

    # Simple elasticity assumption
    price_change_pct = (new_price - base_price) / base_price
    demand_factor = max(0, 1 - 1.5 * price_change_pct)  # elasticity = 1.5

    base_qty = len(product_df)
    new_qty = max(0, int(base_qty * demand_factor))

    # Revenue calculations
    base_revenue = base_qty * base_price
    new_revenue = new_qty * new_price

    st.subheader("Revenue Impact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base Revenue", f"${base_revenue:,.2f}")
    c2.metric("New Revenue", f"${new_revenue:,.2f}")
    c3.metric("Δ Revenue", f"${new_revenue - base_revenue:,.2f}")

    # Plot demand curve
    st.subheader("Demand Curve Simulation")

    prices = np.linspace(base_price * 0.5, base_price * 2.0, 30)
    quantities = [max(0, int(base_qty * (1 - 1.5 * ((p - base_price) / base_price)))) for p in prices]
    revenues = prices * quantities

    fig = px.line(
        x=prices, y=revenues,
        labels={"x": "Price", "y": "Revenue"},
        title="Simulated Price vs Revenue Curve",
    )
    st.plotly_chart(fig, use_container_width=True)



def page_data_dictionary():
    st.title("Data Dictionary")
    st.caption("Plain-English definitions for all key fields and metrics.")

    st.subheader("Transaction-Level Fields (clean_df.csv)")
    st.markdown(
        """
        - **InvoiceNo** – Unique ID for each transaction/invoice  
        - **StockCode** – Product code (SKU)  
        - **Description** – Product name  
        - **Quantity** – Units bought in that line item  
        - **InvoiceDate** – Timestamp of the purchase  
        - **UnitPrice** – Price per unit  
        - **Customer ID** – Unique customer identifier  
        - **Country** – Customer country  
        """
    )

    st.subheader("Customer RFM Features")
    st.markdown(
        """
        - **Recency** – Days since the customer’s last purchase (lower = more recent)  
        - **Frequency** – Number of invoices (higher = more active)  
        - **Monetary** – Total spend in the feature window (higher = more valuable)  
        - **Segment / SegmentName** – Cluster label from RFM (e.g. “Champions”, “At Risk”)  
        """
    )

    st.subheader("Model Outputs")
    st.markdown(
        """
        - **future_spend_3m** – Actual spend in the 3-month target window (training only)  
        - **predicted_clv** – Model-predicted spend for the next 3 months  
        - **purchase_next_3m** – 1/0 flag: did the customer buy again? (training only)  
        - **purchase_probability** – Probability of purchase in the next 3 months  
        - **combined_score** – `predicted_clv × purchase_probability`, used to rank campaign targets  
        """
    )

    st.subheader("Evaluation Metrics")
    st.markdown(
        """
        - **MAE (Mean Absolute Error)** – Average absolute difference between predicted and actual CLV  
        - **RMSE (Root Mean Squared Error)** – Similar to MAE but penalizes large errors more  
        - **Accuracy** – Fraction of customers correctly predicted as buyers/non-buyers  
        - **ROC–AUC** – Measures how well the model ranks likely buyers above non-buyers  
        """
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    # First: check that sklearn and lightgbm are available
    if not SKLEARN_AVAILABLE or not LGB_AVAILABLE:
        st.title("Retail AI Growth Engine")
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn (sklearn) is not installed in this environment.")
            st.code("pip install scikit-learn")
        if not LGB_AVAILABLE:
            st.error("lightgbm is not installed in this environment.")
            st.code("pip install lightgbm")
        return

    # Try loading data & models; if anything fails, show it in the UI
    try:
        clean_tx = load_clean_transactions()
        master_df = build_master_customer_table()
    except Exception as e:
        st.title("Retail AI Growth Engine")
        st.error("Startup error while loading data or models:")
        st.code(str(e))
        st.info(
            "Check that these files exist in the same folder as app.py:\n"
            "- clean_df.csv\n- rfm_segmented_with_names.csv\n"
            "- clv_training_data.csv\n- propensity_training_data.csv\n"
            "- clv_model.txt\n- propensity_model.txt"
        )
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Customer Explorer",
            "Segments",
            "Campaign Designer",
            "Model Diagnostics",
            "Wire Pricing Lab",
            "Data Dictionary",
        ],
    )

    if page == "Overview":
        page_overview(master_df, clean_tx)
    elif page == "Customer Explorer":
        page_customer_explorer(master_df)
    elif page == "Segments":
        page_segments(master_df)
    elif page == "Campaign Designer":
        page_campaign_designer(master_df)
    elif page == "Model Diagnostics":
        page_model_diagnostics()
    elif page == "Wire Pricing Lab":
        page_wire_pricing_lab(clean_tx)

    elif page == "Data Dictionary":
        page_data_dictionary()



if __name__ == "__main__":
    main()
