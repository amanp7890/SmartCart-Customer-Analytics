# SmartCart — Customer Segmentation & Marketing Intelligence

A portfolio-ready customer analytics project that converts customer demographics, spending patterns, purchase channels, and engagement into actionable segments.

## Business problem
A retailer needs to understand **who its customers are, how they shop, what they spend on, and how marketing can be tailored**. SmartCart explores customer-level patterns and creates segments that can support differentiated campaign strategies.

## What is included
- Reusable feature-engineering and segmentation pipeline (`src/segmentation.py`)
- Interactive Streamlit dashboard (`app.py`)
- Reproducible exploratory notebook (`SmartCart_Analysis.ipynb`)
- Customer segment profiles and clustering evaluation
- CSV exports for profiles and filtered customers

## Key improvements over a basic clustering notebook
1. **Leakage-aware features:** campaign `Response` and `Complain` are not used to form segments; response rate is evaluated afterward.
2. **Reusable preprocessing:** median imputation, 1st–99th percentile clipping for model inputs, robust scaling, and one-hot encoding are handled together.
3. **Full feature-space clustering:** K-Means clusters the preprocessed feature matrix; PCA is only for visualization.
4. **Model evaluation:** compare silhouette, Davies–Bouldin, Calinski–Harabasz, and inertia across candidate K values.
5. **Business interpretation:** profile segments using spending, income, purchase channels, recency, and campaign response.
6. **Interactive dashboard:** explore segments, compare channel behavior, and export results.

## Dataset
The included `data/smartcart_customers.csv` contains 2,240 rows and 22 source columns. It includes income, education, household, purchase amounts by product category, purchase channels, recency, complaints, and campaign response. The notebook and app derive additional features.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

To run the notebook:

```bash
jupyter lab SmartCart_Analysis.ipynb
```

## Suggested workflow
1. Validate source data and inspect missingness.
2. Engineer customer age, tenure, total spend, total purchases, channel shares, and household features.
3. Fit preprocessing on numeric and categorical inputs.
4. Compare candidate K values with multiple metrics.
5. Review segment profiles and choose a K that is both statistically reasonable and actionable.
6. Test targeted campaigns and measure incremental lift with a holdout group before claiming business impact.

## Important limitations
- Clustering is exploratory and does not prove that a segment will respond to a campaign.
- The best K depends on metric trade-offs, stability, and business usefulness; do not choose it from one score alone.
- Campaign response differences are descriptive associations, not causal effects.
- Customer age is derived from birth year and the dataset's latest join date year, not an exact date of birth.
- The included dataset is a customer marketing dataset; do not describe it as live transactional data.

## Project structure
```text
SmartCart_Pro/
├── app.py
├── SmartCart_Analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── smartcart_customers.csv
└── src/
    ├── __init__.py
    └── segmentation.py
```

## Resume bullet (edit to reflect what you personally implemented)
**SmartCart Customer Segmentation & Marketing Intelligence** — Built a customer analytics workflow using Python, Pandas, and scikit-learn to engineer spending and channel features, segment customers with K-Means, evaluate cluster quality with multiple metrics, and surface interpretable profiles through a Streamlit dashboard.
