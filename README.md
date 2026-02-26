# Pandas Data Pipeline Project

## Overview

This project demonstrates practical data analysis using **Pandas** and **NumPy** on historical Bitcoin datasets (Coinbase and Bitstamp).  

The objective is to clean, transform, manipulate, aggregate, and visualize time-series financial data while following strict Python style and documentation standards.

---

## Environment

- Python 3.9
- numpy==1.25.2
- pandas==2.2.2
- matplotlib
- Ubuntu 20.04 LTS
- pycodestyle compliant

Dependencies are locked via `requirements.txt`.

---

## Key Concepts Covered

- Creating DataFrames from NumPy arrays and dictionaries
- Loading CSV data into Pandas
- Indexing and slicing DataFrames
- Sorting and filtering data
- Handling missing values
- MultiIndex and hierarchical indexing
- Concatenation and merging
- Statistical analysis with `describe()`
- Time-series resampling
- Data visualization with Matplotlib

---

## Data Processing Workflow

The final pipeline performs:

1. Data loading
2. Column renaming
3. Timestamp → Datetime conversion
4. Indexing on time
5. Missing value handling
6. Daily resampling and aggregation
7. Visualization of 2017+ Bitcoin trends

---

## Final Aggregation Logic (Daily)

- High → max  
- Low → min  
- Open → mean  
- Close → mean  
- Volume_(BTC) → sum  
- Volume_(Currency) → sum  

---

## Outcome

The final visualization produces a daily Bitcoin price trend starting from 2017, demonstrating real-world time-series preprocessing and aggregation techniques.

---

## Learning Outcome

After completing this project, you should confidently understand:

- What Pandas is
- How to build and manipulate DataFrames
- Time-series indexing and resampling
- Cleaning financial datasets
- Producing aggregated analytical outputs
- Visualizing structured data

---

## Author

Christopher – Atlas Machine Learning

