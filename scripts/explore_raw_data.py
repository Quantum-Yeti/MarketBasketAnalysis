import random

import pandas as pd

orders = pd.read_csv("../data/raw/olist_orders_dataset.csv")

print(orders.head())
print(orders.info())
print(orders.describe())

# Missing values
print("\nMissing values: ")
print(orders.isnull().sum())

# Duplicate rows
print("\nDuplicate rows: ")
print(orders.duplicated().sum())

# Order status count
print("\nOrder status counts: ")
for status, count in orders["order_status"].value_counts().items():
    print(f"{status}: {count}")

# Date range
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
print("\nEarliest purchase date: ")
print(orders['order_purchase_timestamp'].min())

print("\nLatest purchase date: ")
print(orders['order_purchase_timestamp'].max())

# Sample random orders
sample_size = random.randint(1, 100)
sampled_orders = orders.sample(n=sample_size).reset_index(drop=True)
print("\nRandom order sample size: ")
print(sample_size)
print(sampled_orders)
