import pandas as pd
import os

def clean_data(data):

    # Merge prior orders and train orders
    orders = data['orders']
    order_products = pd.concat([data['order_products_prior'],data['order_products_train']], ignore_index=True)

    # Merge with products
    merged = order_products.merge(data['products'], on='product_id', how='left')
    merged = merged.merge(data['aisles'], on='aisle_id', how='left')
    merged = merged.merge(data['departments'], on='department_id', how='left')

    # Merge with orders
    merged = merged.merge(orders, on='order_id', how='left')

    # Drop missing data
    merged = merged.dropna(subset=['user_id', 'product_id','order_number'])

    # Feature engineering - total items per order/total orders  per user
    merged['total_items_in_order'] = merged.groupby('order_id')['order_id'].transform('count')
    merged['total_orders_by_user'] = merged.groupby('user_id')['order_number'].transform('max')

    # Reset the index
    merged = merged.reset_index(drop=True)

    # Return merged data
    return merged

def save_clean_data(df, filename="clean_data.csv"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clean", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Cleaned data saved to {}".format(path))