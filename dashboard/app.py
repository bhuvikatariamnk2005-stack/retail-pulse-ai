import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import joblib
import os

from reportlab.pdfgen import canvas
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from prophet import Prophet


st.set_page_config(page_title="Retail Pulse AI", layout="wide")

df = pd.read_csv("data/retail_pulse_cleaned.csv")

np.random.seed(42)

df["customer_id"] = np.random.randint(1000, 2000, size=len(df))
df["order_date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
df["order_date"] = pd.to_datetime(df["order_date"])


st.sidebar.title("Filters")

region = st.sidebar.selectbox("Select Region", df["region"].unique())
filtered_df = df[df["region"] == region]

category = st.sidebar.selectbox("Select Category", filtered_df["category"].unique())
filtered_df = filtered_df[filtered_df["category"] == category]

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Dashboard Info")

st.sidebar.info("""
Retail Pulse Analytics Dashboard  
Built using Streamlit + Pandas + Plotly  

This dashboard allows you to:
- Filter data by Region
- Filter data by Category
- Analyze Sales, Profit & Orders
- View interactive charts
- Download filtered dataset
""")

st.sidebar.markdown("---")
st.sidebar.success("✔️ Dashboard Loaded Successfully")

st.title("📊 Retail Pulse AI Dashboard")

st.markdown("""
### Retail Analytics Dashboard
This dashboard helps analyze sales, profit, orders, and product performance
across different regions and categories.
""")

st.markdown("---")

total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
total_orders = len(filtered_df)
avg_order_value = total_sales / total_orders if total_orders else 0
profit_margin = (total_profit / total_sales) * 100 if total_sales else 0


snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg({
    "order_date": lambda x: (snapshot_date - x.max()).days,
    "sales": "sum",
    "customer_id": "count"
})

rfm.rename(columns={
    "order_date": "recency",
    "customer_id": "frequency",
    "sales": "monetary"
}, inplace=True)

rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4,3,2,1])
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M_score"] = pd.qcut(rfm["monetary"], 4, labels=[1,2,3,4])

rfm["RFM_Score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)


model_path = "churn_model.pkl"

X = rfm[["recency", "frequency", "monetary"]]

median_sales = rfm["monetary"].median()
y = np.where(rfm["monetary"] < median_sales, 1, 0)

if os.path.exists(model_path):
    xgb_model = joblib.load(model_path)
else:
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.fit(X, y)
    joblib.dump(xgb_model, model_path)

rfm["churn_pred"] = xgb_model.predict(X)


ts_df = df.groupby("order_date")["sales"].sum().reset_index()
ts_df.columns = ["ds", "y"]

prophet_model = Prophet()
prophet_model.fit(ts_df)

future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)


st.markdown("---")

total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
total_orders = len(filtered_df)
average_order_value = total_sales / total_orders
profit_margin = (total_profit / total_sales) * 100


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview",
    "🏙️ City Analysis",
    "📦 Product Analysis",
    "📁 Data",
    "📦 Inventory Insights",
    "📈 Executive Analytics",
    "👥 Customer Segmentation",
    "📉 Forecasting"
])

with tab1:
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Total Profit", f"${total_profit:,.0f}")
    col3.metric("Total Orders", total_orders)
    col4.metric("Avg Order Value", f"${average_order_value:.2f}")
    col5.metric("Profit Margin %", f"{profit_margin:.2f}%")

    st.markdown("---")

    st.subheader("Sales by Category")
    st.bar_chart(filtered_df.groupby("category")["sales"].sum())

    st.subheader("Profit by Region")
    st.bar_chart(filtered_df.groupby("region")["profit"].sum())

    st.subheader("🧠 Executive Summary")

    top_city = filtered_df.groupby("city")["sales"].sum().idxmax()
    top_product = filtered_df.groupby("sub-category")["sales"].sum().idxmax()
    loss_making = filtered_df[filtered_df["profit"] < 0]

    profit_status = "Profitable" if total_profit > 0 else "Loss"

    st.info(f"""
📊 Key Insights:

* Top Performing City: {top_city}  
* Best Selling Product Sub-Category: {top_product}  
* Total Profit Status: {profit_status}  
* Loss Transactions Count: {len(loss_making)}  
* Profit Margin: {profit_margin:.2f}%
""")

    st.subheader("⚠️ Smart Alerts & Insights")

    if total_profit < 0:
        st.error("⚠️ Warning: Overall profit is negative.")
    elif profit_margin < 10:
        st.warning("⚠️ Low Profit Margin detected.")
    else:
        st.success("✅ Healthy Profit Performance")

    if total_sales > 1000000:
        st.info("📈 High Sales Performance detected.")

    avg_discount = filtered_df["discount"].mean()

    if avg_discount > 0.3:
        st.warning("⚠️ High discount usage may affect profit.")
    else:
        st.success("💡 Discount levels are controlled.")

    st.subheader("📝 Analyst Note")

    st.info(f"""
* Sales are driven by a few major cities.  
* Some sub-categories dominate revenue.  
* Discount strategy directly impacts profitability.  
* Overall performance: {'Strong' if total_profit > 0 else 'Needs Improvement'}.
""")

with tab2:
    st.subheader("Top 10 Cities by Sales")

    city_sales = filtered_df.groupby("city")["sales"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(city_sales)

    st.subheader("Top 5 Cities by Profit")

    city_profit = filtered_df.groupby("city")["profit"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(city_profit)

    st.subheader("Comparison Insight")

    region_sales = df.groupby("region")["sales"].sum()

    top_region = region_sales.idxmax()
    lowest_region = region_sales.idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🏆 Top Region: {top_region}")

    with col2:
        st.error(f"📉 Lowest Region: {lowest_region}")


with tab3:
    st.subheader("📦 Sales Distribution by Category")

    category_share = filtered_df.groupby("category")["sales"].sum().reset_index()

    fig = px.pie(
        category_share,
        values="sales",
        names="category",
        title="Sales Distribution by Category"
    )
    st.plotly_chart(fig, use_container_width=True, key="tab3_fig1")

    st.subheader("📊 Sales by Sub-Category")

    subcategory_sales = filtered_df.groupby("sub-category")["sales"].sum().sort_values(ascending=False)
    st.bar_chart(subcategory_sales)

    st.subheader("🏆 Top 5 Profitable Sub-Categories")

    top_profit = filtered_df.groupby("sub-category")["profit"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_profit)

    st.subheader("💸 Discount vs Profit Impact")

    fig2 = px.scatter(
        filtered_df,
        x="discount",
        y="profit",
        color="category",
        size="sales",
        title="Discount vs Profit Relationship"
    )
    st.plotly_chart(fig2, use_container_width=True, key="tab3_fig2")

    st.subheader("🧠 Product Insights")

    best_cat = filtered_df.groupby("category")["sales"].sum().idxmax()
    worst_cat = filtered_df.groupby("category")["profit"].sum().idxmin()

    st.info(f"""
    📈 Best Performing Category: {best_cat}  
    ⚠ Weak Profit Category: {worst_cat}  
    💡 Insight: High discounts may be affecting profit in certain sub-categories.
    """)

with tab4:

    st.subheader("📁 Filtered Dataset")

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    st.subheader("⬇️ Download Data")

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "filtered_data.csv",
        "text/csv"
    )

    st.markdown("---")

    st.subheader("📥 Business Report Download")

    report_text = f"""
RETAIL PULSE BUSINESS REPORT

Region: {region}
Category: {category}

Total Sales: {total_sales}
Total Profit: {total_profit}
Orders: {total_orders}
Profit Margin: {profit_margin:.2f}%

Top City: {filtered_df.groupby("city")["sales"].sum().idxmax()}
Top Product: {filtered_df.groupby("sub-category")["sales"].sum().idxmax()}
"""

    st.download_button(
        "Download Report (TXT)",
        report_text,
        "business_report.txt",
        "text/plain"
    )

    st.markdown("---")

    st.subheader("📄 PDF Export")

    generate_pdf = st.button("Generate PDF Report")

    if generate_pdf:

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)

        y = 800

        p.drawString(100, y, "Retail Pulse Business Report")
        y -= 30

        p.drawString(100, y, f"Region: {region}")
        y -= 20

        p.drawString(100, y, f"Category: {category}")
        y -= 20

        p.drawString(100, y, f"Total Sales: ${total_sales:,.2f}")
        y -= 20

        p.drawString(100, y, f"Total Profit: ${total_profit:,.2f}")
        y -= 20

        p.drawString(100, y, f"Profit Margin: {profit_margin:.2f}%")
        y -= 20

        p.drawString(100, y, f"Top City: {top_city}")
        y -= 20

        p.drawString(100, y, f"Top Product: {top_product}")
        y -= 40

        p.drawString(100, y, "Key Recommendation:")
        y -= 20

        p.drawString(
            100,
            y,
            "Focus investment on high-performing products and regions."
        )

        p.save()
        buffer.seek(0)

        st.download_button(
            "⬇️ Download PDF",
            buffer,
            "business_report.pdf",
            "application/pdf"
        )
    with tab5:
        st.subheader("📦 Inventory Recommendations")

    inventory_df = filtered_df.groupby("sub-category").agg({
        "sales": "sum",
        "quantity": "sum",
        "profit": "sum"
    }).reset_index()

    st.dataframe(inventory_df, use_container_width=True)

    st.subheader("⚡ Stock Demand Analysis")

    inventory_df["demand_level"] = np.where(
        inventory_df["quantity"] > inventory_df["quantity"].mean(),
        "High Demand",
        "Normal Demand"
    )

    st.write(inventory_df["demand_level"].value_counts())

    st.subheader("🏆 Top Revenue Products")

    top_products = inventory_df.sort_values("sales", ascending=False).head(7)
    st.bar_chart(top_products.set_index("sub-category")["sales"])

    st.subheader("⚠ Low Profit Products")

    low_profit = inventory_df.sort_values("profit").head(7)
    st.bar_chart(low_profit.set_index("sub-category")["profit"])

    st.subheader("🧠 Inventory Insights")

    st.info("""
    • High demand items should be restocked frequently  
    • Low profit items may need pricing review  
    • Some products drive sales but not profit  
    • Inventory optimization can improve margins significantly  
    """)

with tab6:

    st.subheader("🌍 Regional Performance Ranking")

    region_perf = (
        df.groupby("region")[["sales", "profit"]]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
    )

    st.dataframe(region_perf, use_container_width=True)

    if not region_perf.empty:

        best_region = region_perf.iloc[0]
        worst_region = region_perf.iloc[-1]

        st.success(
            f"🏆 Best Region: {best_region['region']} "
            f"with Sales ${best_region['sales']:,.0f}"
        )

        st.error(
            f"📉 Lowest Region: {worst_region['region']} "
            f"with Sales ${worst_region['sales']:,.0f}"
        )

    fig = px.bar(
        region_perf,
        x="region",
        y="profit",
        title="Profit by Region"
    )

    st.plotly_chart(fig, use_container_width=True, key="tab6_fig1")

    st.subheader("📊 Sales Share by Region")

    region_sales = (
        df.groupby("region")["sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.pie(
        region_sales,
        values="sales",
        names="region",
        title="Regional Sales Distribution"
    )
st.plotly_chart(fig2, use_container_width=True, key="tab6_fig2")
st.subheader("🧠 Management Recommendations")
st.info("""
    1. Increase focus on high-performing regions.
    2. Reduce discounts on low-margin products.
    3. Improve inventory for high-demand categories.
    4. Monitor loss-making sub-categories.
    5. Reallocate marketing spend toward profitable regions.
    """)
with tab7:
        st.subheader("👥 Customer Segmentation (K-Means)")
        customer_df = df.groupby("customer_id").agg({
        "sales": "sum",
        "profit": "sum",
        "quantity": "sum"
    }).reset_index()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(
        customer_df[["sales", "profit", "quantity"]]
    )
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        customer_df["cluster"] = kmeans.fit_predict(scaled_data)
        st.subheader("📊 Customer Clusters Visualization")
        
        fig = px.scatter(
        customer_df,
        x="sales",
        y="profit",
        color="cluster",
        size="quantity",
        title="Customer Segments (K-Means Clustering)"
    )
        st.plotly_chart(fig, use_container_width=True, key="tab7_fig1")
        st.subheader("⚠️ Customer Churn Risk Analysis")
        median_sales = customer_df["sales"].median()
        customer_df["churn_risk"] = np.where(
        customer_df["sales"] < median_sales,
        "High Risk",
        "Low Risk"
    )
        fig2 = px.pie(
        customer_df,
        names="churn_risk",
        title="Customer Churn Risk Distribution"
    )
        st.plotly_chart(fig2, use_container_width=True, key="tab7_fig2")
        st.subheader("🚨 High Risk Customers")
        
        st.dataframe(
        customer_df[customer_df["churn_risk"] == "High Risk"]
        .sort_values("sales")
        .head(10)
    )
        st.subheader("🧠 Customer Insights")
        st.info("""
    • High-value customers contribute majority revenue  
    • Low sales customers have higher churn risk  
    • Cluster analysis helps identify customer groups  
    • Retention strategies should target low engagement users  
    """)
        
        with tab8:
            st.subheader("📉 Sales Forecasting (Prophet Model)")
            from prophet import Prophet
            
            forecast_df = df.groupby("order_date")["sales"].sum().reset_index()
            forecast_df.columns = ["ds", "y"]
            
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
            st.write("Training forecasting model...")
            
            model = Prophet()
            
            model.fit(forecast_df)
            
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            st.subheader("📊 Forecast Plot")
            
            fig1 = model.plot(forecast)
            
            st.pyplot(fig1)
            st.subheader("📈 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            st.subheader("🔮 Next 7 Days Forecast")
            st.dataframe(forecast[["ds", "yhat"]].tail(7))