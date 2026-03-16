# Instacart Market Basket Analysis
## E-Commerce BI Project

---

### Project Overview
This project analyzes the Instacart dataset to uncover patterns in customer purchasing behavior, product reorders, and basket composition. Using the relational data from multiple tables, theproject demonstrates advanced data cleaning, feature engineering, exploratory data analysis, and predictive modeling.

### Dataset
The dataset is sourced from ![Kaggle: Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)


| Table                     | Description                                                     |
|---------------------------|-----------------------------------------------------------------|
| orders.csv                | Customer orders, timestamps, and user info                      |
| order_products__prior.csv | Products purchased in prior orders                              |
| order_products__train.csv | Products purchased in train set orders (for reorder prediction) |
| products.csv              | Product IDs, names, aisle and department IDs                    |
| aisles.csv                | Aisle names and IDs                                             |
| departments.csv           | Department names and IDs                                        |

### Project Goals

- Clean and merge multiple relational tables to create an integrated dataset.
- Conduct Exploratory Data Analysis (EDA) to understand customer purchasing habits.
- Generate insights  into product reorders,basket composition, and departmental trends.
- Build a predictive model for product reordering behavior.
- Develop interactive dashboards for visualization and decision support (Power BI/Tableau).

### Tools & Technologies

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- SQL (joins, aggregations, feature engineering)
- Jupyter Notebook for interactive exploration
- Power BI (or Tableau) for dashboards

### Key Features / Deliverables

- Data cleaning scripts for handling missing values, duplicates, and inconsistent timestamps.
- Merged and enriched datasets ready for analysis.
- Correlation matrices and summary statistics for key features.
- Predictive modeling to identify reorder likelihood for products.
- Interactive dashboards highlighting product, aisle, and department trends over time.

### Project Structure
```project/
├── data/
│ ├── raw/ # Original Instacart CSV files
│ └── clean/ # Cleaned and merged dataset(s)
├── notebooks/ # Jupyter notebooks for analysis
├── scripts/ # Python scripts for ETL and modeling
├── reports/ # Charts, plots, and dashboard screenshots
└── README.md # Project documentation
```



