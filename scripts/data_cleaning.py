import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def clean_data(data):
    # Merge prior orders and train orders
    orders = data['orders']
    order_products = pd.concat([data['order_products_prior'], data['order_products_train']], ignore_index=True)

    # Normalize column names for safety
    for df in [order_products, data['products'], data['aisles'], data['departments'], orders]:
        df.columns = df.columns.str.strip().str.lower()

    # Merge with products, aisles, departments
    merged = order_products.merge(data['products'], on='product_id', how='left')
    merged = merged.merge(data['aisles'], on='aisle_id', how='left')
    merged = merged.merge(data['departments'], on='department_id', how='left')

    # Merge with orders
    merged = merged.merge(orders, on='order_id', how='left')

    # Drop missing data
    merged = merged.dropna(subset=['user_id', 'product_id', 'order_number'])

    # Drop duplicates
    merged = merged.drop_duplicates()

    # Convert relevant data to categorical
    categorical_columns = ['product_name', 'aisle', 'department']
    for column in categorical_columns:
        if column in merged.columns:
            merged[column] = merged[column].astype('category')

    # Feature engineering
    merged['total_items_in_order'] = merged.groupby('order_id')['order_id'].transform('count')
    merged['total_orders_by_user'] = merged.groupby('user_id')['order_number'].transform('max')

    # Reset index
    merged = merged.reset_index(drop=True)

    print(f"Merged data shape: {merged.shape}")
    return merged

def save_clean_data(df, filename="clean_data.csv"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clean", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}")

def export_powerbi_tables(df):
    # Product demand
    product_stats = df.groupby(
        ['product_id', 'product_name']
    ).agg(
        total_purchases=('product_id', 'count'),
        reorder_probability=('reordered', 'mean')
    ).reset_index()

    # User stats
    user_stats = df.groupby('user_id').agg(
        total_orders=('order_number', 'max'),
        total_items=('product_id', 'count')
    ).reset_index()

    # Department demand
    department_stats = df.groupby('department').agg(
        total_orders=('order_id', 'count'),
        avg_reorder=('reordered', 'mean')
    ).reset_index()

    product_stats.to_csv("../data/product_stats.csv", index=False)
    user_stats.to_csv("../data/user_stats.csv", index=False)
    department_stats.to_csv("../data//department_stats.csv", index=False)

    print("Power BI tables exported.")

def generate_apriori(df, sample_orders=50000, top_products=200, min_support=0.01, min_confidence=0.3):
    """
    Memory-efficient Apriori for large datasets reduced to patterns for bananas.
    - df: cleaned dataframe with 'order_id' and 'product_name'
    - sample_orders: number of orders to sample for analysis
    - top_products: only consider top N products by purchase frequency
    - min_support: minimum support for frequent itemsets
    - min_confidence: minimum confidence for association rules
    """

    # Sample orders (memory-friendly)
    unique_orders = df['order_id'].drop_duplicates()
    sampled_orders = unique_orders.sample(n=min(sample_orders, len(unique_orders)), random_state=42)
    df_sample = df[df['order_id'].isin(sampled_orders)]

    # Keep only top products
    top_prod_list = df_sample['product_name'].value_counts().head(top_products).index
    df_sample = df_sample[df_sample['product_name'].isin(top_prod_list)]

    # Convert to list of transactions
    transactions = df_sample.groupby('order_id')['product_name'].apply(list).tolist()

    # TransactionEncoder with sparse matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)  # Sparse = low memory
    basket_sparse = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    # Frequent itemsets
    frequent_itemsets = apriori(basket_sparse, min_support=min_support, use_colnames=True)

    # Association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

    # Clean rules for Power BI
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    pbi_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']] \
        .sort_values(by='lift', ascending=False)

    # Export CSV for Power BI
    output_path = "../data/apriori_rules.csv"
    pbi_rules.to_csv(output_path, index=False)
    print(f"Apriori rules exported to {output_path}")

    return pbi_rules