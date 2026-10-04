Logistics Data Analyst Internship Project
Internship: Yuva Intern
Duration: 24 Aug 2026 – 21 Sep 2026
Author: Hari Sai Yugesh  
Overview
This project demonstrates a complete end-to-end logistics analytics workflow covering:
- Strategic planning
- Data preprocessing
- Exploratory data analysis (EDA)
- Predictive modeling
- Operational optimization
The objective is to show how data-driven decision-making improves logistics performance, including delivery efficiency, cost optimization, and resource allocation.
Key Highlights
- Simulated dataset of 8,000+ logistics orders
- Built predictive models for delivery time forecasting
- Identified key bottlenecks (West zone, Same Day delivery)
- Achieved approximately 0.46 days (~11 hours) prediction accuracy
- Proposed practical optimization strategies based on insights
Project Structure
- Week1_Strategic_Planning_Report.docx
- Week2_Data_Cleaning_Report.docx
- Week3_Data_Analysis_Visualization_Report.docx
- Week4_Predictive_Modeling_Optimization_Report.docx
- pipeline.py
- analysis.py
- week4_model.py
- charts/
- datasets/
Week 1 — Strategic Planning and Data Exploration
A realistic logistics scenario was defined involving:
- Late deliveries
- High delivery costs
- Inventory imbalance
Key Performance Indicators (KPIs)
- OTIF (On-Time In-Full) rate
- Cost per delivery
- Inventory turnover
- Average lead time
- Vehicle capacity utilization
- Stock-out rate
Additional Work
- Identified public datasets (DataCo, Olist, Amazon routing data)
- Mapped data science techniques:
  - Regression
  - Classification
  - Clustering
  - Forecasting
  - Optimization
- Designed a 7-stage analytics roadmap
- Included Python pseudocode for each stage
Week 2 — Data Collection, Cleaning, and Preprocessing
The file pipeline.py simulates and cleans a logistics dataset of 5,120 records.
Data Issues Introduced
- Missing values
- Inconsistent categories
- Outliers
- Duplicate records
Techniques Applied
- Removed duplicates using order_id
- Converted data types using pandas
- Standardized categorical values
- Applied rule-based validation for incorrect values
- Used IQR method for outlier detection and capping
- Handled missing values:
  - Median for numerical columns
  - "Unknown" for categorical columns
- Feature engineering:
  - is_late
  - fill_rate
  - cost_per_km
  - order_month
- Data normalization:
  - Min-max scaling
  - Z-score normalization
- Validation using assertions
Results
- 5,120 rows reduced to 5,000 rows
- 974 missing values reduced to 0
- 13 category variations reduced to 4
- Delivery cost distortion reduced (~45%)
Week 3 — Advanced Data Analysis and Visualization
The file analysis.py simulates 8,000 logistics orders (Jan–Jun 2026) and performs EDA.
Visualizations Created
- Daily order trends (line chart)
- Cost and distance distributions (histograms)
- Cost by shipping mode (box plot)
- Distance vs cost (scatter plot)
- Correlation heatmap
- Late rate by zone and hour
- Late rate by zone and shipping mode
- Monthly OTIF vs order volume
Key Insights
- 21.6% orders were late → OTIF = 75%
- Distance is the strongest cost driver (correlation ≈ 0.80)
- West zone has the highest delays (42.1%)
- Central zone has the lowest delays (11.5%)
- March demand surge reduced OTIF to 66.3%
- Same Day delivery:
  - Highest cost
  - Lowest reliability (42.5% late)
- Order hour and weight have minimal impact on delays
Recommendations
- Improve operations in the West zone
- Adjust delivery promises based on realistic constraints
- Restrict or reprice Same Day delivery
- Plan resources for peak demand periods
- Reduce fixed cost per delivery
- Implement weekly KPI monitoring
Week 4 — Predictive Modeling and Optimization
The file week4_model.py builds predictive models to forecast delivery time (actual_days).
Problem Definition
Predict delivery time to:
- Improve service reliability
- Enable proactive logistics planning
Features Used
- distance_km
- weight_kg
- load_ratio
- zone
- ship_mode
- order_hour
- weekday
- month
Models Implemented
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
Model Performance
Linear Regression  
- MAE: 0.46 days
- RMSE: 0.58
- R²: 0.49
Random Forest  
- MAE: 0.47 days
- RMSE: 0.59
- R²: 0.47
Gradient Boosting  
- MAE: 0.46 days
- RMSE: 0.58
- R²: 0.49
Final Model Selection
Linear Regression was selected due to:
- Comparable performance
- High interpretability
- Simplicity for real-world deployment
Optimization Strategies
1. Capacity Planning
- Allocate more delivery resources to high-delay zones
- Increase fleet capacity during peak months
2. Service Optimization
- Restrict Same Day delivery for long-distance orders
- Apply eligibility rules using predicted delivery time
3. Route Optimization
- Identify inefficient delivery patterns
- Future scope:
  - Vehicle Routing Problem (VRP)
  - Google OR-Tools integration
4. Cost Optimization
- Improve vehicle utilization
- Reduce fixed delivery costs
5. Risk-Based Prioritization
- Predict high-risk (late) deliveries
- Prioritize shipments dynamically
Key Outcome
- Prediction accuracy ≈ 0.46 days (~11 hours)
- Enables proactive logistics decisions
- Transforms analytics pipeline into decision-support system
How to Run
Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn  
Run scripts:
python pipeline.py
python analysis.py
python week4_model.py  
Outputs
Week 2
- raw_logistics_orders.csv
- clean_logistics_orders.csv
- stats.json
Week 3
- logistics_orders_week3.csv
- stats3.json
- charts/
Week 4
- week4_test_predictions.csv
- model evaluation results
Tools Used
- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
Final Summary
This project demonstrates the full lifecycle of logistics analytics:
1. Strategic planning
2. Data preprocessing
3. Exploratory analysis
4. Predictive modeling
5. Optimization
It highlights how data analytics improves:
- Delivery performance
- Cost efficiency
- Resource allocation
- Operational decision-making
Transition achieved:
Descriptive Analytics → Predictive Analytics → Prescriptive Analytics  
Contact
LinkedIn: (add your link)
GitHub: (add your repo link)
