from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.segmentation import load_and_engineer, fit_segmentation, evaluate_k_values, make_cluster_profile

st.set_page_config(page_title="SmartCart | Customer Intelligence", page_icon="🛒", layout="wide")
st.title("🛒 SmartCart Customer Intelligence")
st.caption("Customer segmentation • spending behaviour • channel strategy • campaign insights")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload customer CSV (optional)", type=["csv"])
    k = st.slider("Number of customer segments (K)", min_value=2, max_value=8, value=3)
    st.markdown("---")
    st.caption("Segmentation excludes campaign response and complaint outcomes from model inputs.")

data_path = uploaded if uploaded is not None else ROOT / "data" / "smartcart_customers.csv"
try:
    raw = pd.read_csv(data_path) if uploaded is not None else pd.read_csv(data_path)
    customers = load_and_engineer(raw)
    result = fit_segmentation(customers, k=k)
    labeled = customers.copy()
    labeled["Cluster"] = result["labels"]
except Exception as exc:
    st.error(f"Could not process the dataset: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers analyzed", f"{len(labeled):,}")
m2.metric("Average spend", f"₹{labeled['Total_Spending'].mean():,.0f}")
m3.metric("Average income", f"₹{labeled['Income'].mean():,.0f}")
m4.metric("Silhouette score", f"{result['metrics']['silhouette']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Model evaluation", "Customer explorer"])
with tab1:
    left, right = st.columns(2)
    with left:
        fig = px.scatter(labeled, x="Total_Spending", y="Income", color=labeled["Cluster"].astype(str),
                         hover_data=["Age","Total_Purchases","Recency"], title="Income vs total spending",
                         labels={"color":"Segment"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(labeled, x="Total_Spending", color=labeled["Cluster"].astype(str),
                           barmode="overlay", opacity=.7, title="Spending distribution by segment",
                           labels={"color":"Segment"})
        st.plotly_chart(fig, use_container_width=True)
    channel_cols = ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]
    channel = labeled.groupby("Cluster")[channel_cols].mean().reset_index().melt(
        id_vars="Cluster", var_name="Channel", value_name="Average purchases")
    fig = px.bar(channel, x="Cluster", y="Average purchases", color="Channel", barmode="group",
                 title="Average purchase activity by channel")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    profile = make_cluster_profile(labeled, labeled["Cluster"])
    st.subheader("Segment profile")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.download_button("Download segment profile CSV", profile.to_csv(index=False).encode("utf-8"),
                       file_name="smartcart_segment_profile.csv", mime="text/csv")
    st.subheader("Suggested activation ideas")
    for _, row in profile.iterrows():
        cluster = int(row["Cluster"])
        spend, recency = row["Total_Spending"], row["Recency"]
        if spend >= profile["Total_Spending"].median():
            idea = "VIP rewards, early access, and premium product bundles."
        elif recency >= profile["Recency"].median():
            idea = "Win-back campaign, personalized offer, and re-engagement email."
        else:
            idea = "Cross-sell complementary products and test a small targeted incentive."
        st.markdown(f"**Segment {cluster}** — {idea}")

with tab3:
    st.write("Compare multiple K values. Use these metrics alongside business interpretability; no single score determines the best segmentation.")
    with st.spinner("Evaluating candidate segment counts…"):
        evaluation = evaluate_k_values(customers, range(2, 9))
    st.dataframe(evaluation.round(4), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(evaluation, x="k", y="silhouette", markers=True,
                                title="Silhouette score (higher is better)"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(evaluation, x="k", y="davies_bouldin", markers=True,
                                title="Davies–Bouldin index (lower is better)"), use_container_width=True)
    st.info("Select K using metric trade-offs, segment size, stability, and whether each group supports a distinct business action.")

with tab4:
    st.subheader("Filter and inspect customers")
    min_spend, max_spend = int(labeled["Total_Spending"].min()), int(labeled["Total_Spending"].max())
    spend_range = st.slider("Total spending range", min_value=min_spend, max_value=max_spend,
                             value=(min_spend, max_spend))
    selected_cluster = st.multiselect("Segments", sorted(labeled["Cluster"].unique()),
                                      default=sorted(labeled["Cluster"].unique()))
    filtered = labeled[
        labeled["Total_Spending"].between(*spend_range) & labeled["Cluster"].isin(selected_cluster)
    ]
    display_cols = ["ID","Cluster","Age","Income","Total_Spending","Total_Purchases","Recency",
                    "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","Response"]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button("Download filtered customers CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="smartcart_filtered_customers.csv", mime="text/csv")

with st.expander("Data quality and methodology"):
    st.write(f"Rows retained after basic plausibility checks: {len(customers):,}")
    st.write(f"Missing income values after feature engineering: {customers['Income'].isna().sum():,} (imputed in the modeling pipeline).")
    st.write("The model uses standardized numeric features and one-hot encoded customer categories. PCA is used for visual exploration only; clustering is performed on the full preprocessed feature matrix.")
    st.write("Response and complaint fields are excluded from clustering inputs to reduce outcome leakage. Response rate is shown only after segments are assigned.")
