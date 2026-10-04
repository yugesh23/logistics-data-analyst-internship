Logistics Data Analyst Internship
Work completed for the Logistics Data Analyst Internship at Yuva Intern (24 Aug 2026 to 21 Sep 2026).
Author: Hari Sai Yugesh
Contents
Week	Topic	Files
1	Strategic Planning and Data Exploration in Logistics	Week1_Strategic_Planning_Report.docx
2	Data Collection, Cleaning, and Preprocessing	Week2_Data_Cleaning_Report.docx, pipeline.py
3	Advanced Data Analysis and Visualization	Week3_Data_Analysis_Visualization_Report.docx, analysis.py
4	Predictive Modeling and Optimization in Logistics Systems	Week4_Predictive_Modeling_Optimization_Report.docx, week4_model.py


Note on the data
No real company dataset was available, so the data in Weeks 2, 3 and 4 is simulated. The figures below come from running the code on that simulated data and show how the methods work. The same code can be rerun on real datasets such as DataCo Smart Supply Chain or Olist.
Week 1: Strategic Planning
A report that defines a realistic logistics scenario (a regional distributor facing late deliveries, high delivery cost and unbalanced inventory) and covers:
- Six KPIs: OTIF rate, cost per delivery, inventory turnover, average lead time, vehicle capacity utilisation and stock-out rate
- Background research and public data sources (DataCo Smart Supply Chain, Olist, Amazon Last Mile Routing, OpenStreetMap, World Bank LPI)
- How regression, classification, clustering, forecasting and route optimisation apply to logistics
- A seven-stage analysis roadmap with risks and mitigation
- Python code illustrations for each stage
Week 2: Data Collection, Cleaning and Preprocessing
pipeline.py generates a simulated raw logistics dataset (5,120 orders, modelled on public shipping datasets, with deliberately injected quality problems) and then cleans it.
Step	Technique
Duplicates	Drop by key (order_id)
Data types	pd.to_numeric(errors="coerce")
Categories	Strip whitespace, standardise case, merge abbreviations
Impossible values	Rule-based validation, set to missing
Outliers	IQR method with capping and an audit flag column
Missing values	Group median for numeric columns, "Unknown" label for categories
Feature engineering	is_late, fill_rate, cost_per_km, order_month
Scaling	Min-max and z-score normalisation
Validation	Assertions before export


Results
- 5,120 raw rows reduced to 5,000 clean rows (120 duplicates removed)
- 974 missing cells reduced to 0
- 13 shipping-mode spellings reduced to 4
- Raw mean delivery cost was about 45% higher than the cleaned value because of extreme outliers
Week 3: Advanced Data Analysis and Visualization
analysis.py simulates 8,000 orders (January to June 2026, with a festival-season surge and zone differences built in), runs exploratory data analysis, and produces eight charts that are embedded in the report.
Chart	Type	Purpose
Daily order volume	Line chart with 7-day average	Trend and surge detection
Cost and distance	Histograms	Distribution shape and skew
Cost by shipping mode	Box plot	Cost level and spread by service
Distance vs cost	Scatter plot with fitted line	Cost driver and fixed cost component
Correlations	Heat map	Which variables move together
Late rate by zone and hour	Bar and line charts	Where delays occur
Late rate by zone and mode	Two-way heat map	Problem combinations
Monthly orders and OTIF	Bar and line combination	Service level against workload


Key findings (on the simulated data)
- 21.6% of orders were late and OTIF was 75.0%
- Distance is the main cost driver (correlation 0.80, about INR 4.2 per km)
- The West zone is the main bottleneck: 42.1% of orders late against 11.5% in Central
- The March surge dropped monthly OTIF to 66.3%, from roughly 77% in other months
- Same Day is the least reliable service (42.5% late) and the most expensive
- Order hour and shipment weight do not explain lateness
The report ends with six recommendations: fix the West zone, re-set delivery promises, restrict or reprice Same Day, plan for peak periods, reduce fixed cost per delivery, and monitor weekly.
Week 4: Predictive Modeling and Optimization
week4_model.py builds predictive models to forecast delivery time and proposes optimization strategies.
Problem Definition
The objective is to predict delivery time (actual_days) and use these predictions to improve logistics operations.
Dataset and Features
The model uses the cleaned simulated dataset with the following features:
- distance_km
- weight_kg
- load_ratio
- zone
- ship_mode
- order_hour
- weekday
- month
Models Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
Model Performance
- Linear Regression: MAE ≈ 0.46 days, RMSE ≈ 0.58, R² ≈ 0.49
- Random Forest: MAE ≈ 0.47 days, RMSE ≈ 0.59, R² ≈ 0.47
- Gradient Boosting: MAE ≈ 0.46 days, RMSE ≈ 0.58, R² ≈ 0.49
Linear Regression was selected as the final model due to its balance of performance and interpretability.
Optimization Strategies
- Capacity Planning: allocate more resources to high-delay zones and increase fleet during peak periods
- Service Optimization: restrict Same Day delivery for long-distance orders and apply prediction-based rules
- Route Optimization: identify inefficient routes and propose future VRP-based improvements
- Cost Optimization: improve vehicle utilization and reduce fixed delivery costs
- Risk-Based Prioritization: flag high-risk deliveries using predictions
Outcome
- Prediction error is approximately 0.46 days (~11 hours)
- Enables proactive logistics decision-making
- Extends analysis into predictive and prescriptive analytics
How to run
pip install pandas numpy scikit-learn matplotlib seaborn
python pipeline.py    # Week 2
python analysis.py    # Week 3
python week4_model.py # Week 4  
pipeline.py writes:
- raw_logistics_orders.csv: the messy simulated input
- clean_logistics_orders.csv: the cleaned and scaled output
- stats.json: before and after statistics
analysis.py writes:
- logistics_orders_week3.csv: the simulated Week 3 dataset
- stats3.json: EDA statistics
- charts/: the eight charts as PNG files
week4_model.py writes:
- week4_test_predictions.csv: predicted vs actual values
- model evaluation metrics
Tools
Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn
