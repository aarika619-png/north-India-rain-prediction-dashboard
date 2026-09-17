# Dashboard for Rain Prediction in North India

## Project Overview

This project develops an interactive dashboard for analyzing rainfall patterns and predicting rainfall in North India.

The project combines data cleaning, exploratory data analysis, machine learning, and interactive visualization.

## Objectives

- Analyze rainfall patterns across North Indian states and districts.
- Study monthly, seasonal, and yearly rainfall trends.
- Identify areas with high and low rainfall.
- Analyze relationships between rainfall and other weather variables.
- Build a machine learning model for rainfall prediction.
- Develop an interactive dashboard for presenting rainfall analysis and predictions.

## Dataset

The project uses an India weather and rainfall dataset containing weather observations and rainfall-related information.

The dataset was inspected and cleaned before exploratory analysis and predictive modeling.

## Data Processing

The workflow included:

- Data structure and data-type inspection
- Missing-value analysis
- Duplicate-record checking
- Data consistency checks
- Appropriate cleaning of invalid or inconsistent values
- Preservation of valid extreme rainfall observations

## Exploratory Data Analysis

The project analyzes:

- Rainfall by state
- Rainfall by district
- Monthly rainfall patterns
- Seasonal rainfall patterns
- Rainfall trends over time
- Rainfall distribution
- Extreme rainfall events
- Relationships between weather variables

## Machine Learning

A rainfall prediction model was developed using the cleaned dataset.

The modeling workflow includes:

- Feature selection
- Categorical-variable encoding
- Time-aware train/test splitting where dates are available
- Model training and comparison
- Model evaluation
- Feature-importance analysis

## Dashboard Features

The interactive dashboard includes:

- Total rainfall KPI
- Average rainfall KPI
- Number of observations
- Highest rainfall value
- Rainfall trends over time
- State-wise rainfall comparison
- Monthly and seasonal analysis
- Rainfall distribution and extreme-event analysis
- Weather-variable relationships
- Interactive rainfall prediction
- Model performance metrics
- Interactive filters

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Excel
- Git
- GitHub

## How to Run

Install the required Python packages and run:

```bash
pip install -r requirements.txt
streamlit run app.py
