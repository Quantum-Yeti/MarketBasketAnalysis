import pandas as pd
import os

from sympy.simplify.simplify import product_simplify

# Handle relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the raw data
orders = pd.read_csv(os.path.join(script_dir, "..", "data", "raw", "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(script_dir, "..", "data", "raw", "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(script_dir, "..", "data", "raw", "olist_products_dataset.csv"))
customers = pd.read_csv(os.path.join(script_dir, "..", "data", "raw", "olist_customers_dataset.csv"))

# Convert timestamps
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# Merge tables
sales = order_items.merge(orders, on='order_id', how='left')
sales =  sales.merge(products, on='product_id', how='left')
sales = sales.merge(customers, on='customer_id', how='left')

# Keep columns that are relevant / Create pandas DataFrame
sales = sales[[
    'order_id',
    'customer_id',
    'seller_id',
    'product_id',
    'price',
    'freight_value',
    'product_category_name',
    'customer_city',
    'customer_state',
    'order_status',
    'order_purchase_timestamp',
]]

# Save and store the cleaned dataset
os.makedirs(os.path.join(script_dir, "..", "data", "clean"), exist_ok=True)
sales.to_csv(os.path.join(script_dir, "..", "data", "clean", "olist_orders_cleaned.csv"), index=False)

print("\nCleaned data saved in {}".format(os.path.join(script_dir, "..", "data", "clean_dataset")))