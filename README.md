# Logistics Data Analyst Internship

Work completed for the Logistics Data Analyst Internship at Yuva Intern (24 Aug 2026 to 21 Sep 2026).

**Author:** Hari Sai Yugesh

## Contents

| Week | Topic | Files |
|------|-------|-------|
| 1 | Strategic Planning and Data Exploration in Logistics | `Week1_Strategic_Planning_Report.docx` |
| 2 | Data Collection, Cleaning, and Preprocessing | `Week2_Data_Cleaning_Report.docx`, `pipeline.py` |

## Week 1: Strategic Planning

A report that defines a realistic logistics scenario (a regional distributor facing late deliveries, high delivery cost and unbalanced inventory) and covers:

- Six KPIs: OTIF rate, cost per delivery, inventory turnover, average lead time, vehicle capacity utilisation and stock-out rate
- Background research and public data sources (DataCo Smart Supply Chain, Olist, Amazon Last Mile Routing, OpenStreetMap, World Bank LPI)
- How regression, classification, clustering, forecasting and route optimisation apply to logistics
- A seven-stage analysis roadmap with risks and mitigation
- Python code illustrations for each stage

## Week 2: Data Collection, Cleaning and Preprocessing

`pipeline.py` generates a **simulated** raw logistics dataset (5,120 orders, modelled on public shipping datasets, with deliberately injected quality problems) and then cleans it.

| Step | Technique |
|------|-----------|
| Duplicates | Drop by key (`order_id`) |
| Data types | `pd.to_numeric(errors="coerce")` |
| Categories | Strip whitespace, standardise case, merge abbreviations |
| Impossible values | Rule-based validation, set to missing |
| Outliers | IQR method with capping and an audit flag column |
| Missing values | Group median for numeric columns, "Unknown" label for categories |
| Feature engineering | `is_late`, `fill_rate`, `cost_per_km`, `order_month` |
| Scaling | Min-max and z-score normalisation |
| Validation | Assertions before export |

### Results (on the simulated data)

- 5,120 raw rows reduced to 5,000 clean rows (120 duplicates removed)
- 974 missing cells reduced to 0
- 13 shipping-mode spellings reduced to 4
- Raw mean delivery cost was about 45% higher than the cleaned value because of extreme outliers

### Note on the data

No real company dataset was available, so the data is simulated. The figures above come from running the pipeline on that simulated data. The same methods apply to real datasets such as DataCo Smart Supply Chain or Olist.

## How to run

```bash
pip install pandas numpy scikit-learn
python pipeline.py
```

This writes:

- `raw_logistics_orders.csv`: the messy simulated input
- `clean_logistics_orders.csv`: the cleaned and scaled output
- `stats.json`: before and after statistics

## Tools

Python 3, pandas, NumPy, scikit-learn
