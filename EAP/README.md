# TechNova Solutions - Employee Attrition Prediction

## Project Overview
Predictive analytics project to identify employees at risk of leaving TechNova Solutions using machine learning techniques.

## Business Objective
- Predict employee churn with target Recall ≥80%
- Identify key drivers of attrition
- Provide actionable recommendations for HR interventions

## Dataset
- **Source:** TechNova employee data
- **Size:** 10,000 employees, 22 features
- **Target Variable:** Churn (0 = Stay, 1 = Leave)
- **Class Distribution:** 79.7% Stay, 20.3% Leave

## Methodology
1. Exploratory Data Analysis (EDA)
2. Feature Engineering (created 12 new features)
3. Model Selection & Training (Random Forest with GridSearchCV)
4. Model Evaluation & Explainability
5. Business Recommendations

## Key Findings
- **Model Performance:** Recall 29.31% (below target of 80%)
- **Top 5 Churn Drivers:**
  1. Satisfaction_Performance_Ratio (8.5%)
  2. Salary (7.9%)
  3. Workload_Indicator (6.5%)
  4. Hours_Per_Project (5.8%)
  5. Manager Feedback Score (5.5%)

## Key Insights
- Weak predictive signals across all features (correlations <0.05)
- Model cannot reliably distinguish between stay/leave employees
- 98.75% of predictions fall in medium confidence range (40-60%)

## Recommendations
- Focus on top 5 risk factors through manual HR assessment
- Implement satisfaction surveys and workload monitoring
- Accelerate career development programs
- **Do not deploy model** until performance improves significantly

## Files
- `TechNova_Attrition_Prediction_<your_id>.ipynb` - Main analysis notebook
- `employee_churn_dataset.csv` - Raw data
- `employee_churn_data_dictionary.csv` - Data dictionary

## Technologies Used
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## How to Run
1. Clone this repository
2. Install required libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open the Jupyter Notebook
4. Run all cells sequentially

## Author
[Your Name] - [Your Student ID]

## Date
October 2025
