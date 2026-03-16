import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


def run_exploration():
    # Load clean data
    df = pd.read_csv("../data/clean/clean_data.csv")

    explorer = DataExplorer(df, sample_size=100000)

    explorer.overview()
    explorer.missing_values()
    explorer.descript_stats()
    explorer.plot_histograms()
    explorer.plot_boxplot()
    explorer.correlation_matrix()

    pivot_user = explorer.pivot_table_summary('user_id')
    pivot_product = explorer.pivot_table_summary('product_name')

    # Linear Regression: 'total_items_in_order' vs 'add_to_cart_order'
    explorer.linear_regression('total_items_in_order', 'add_to_cart_order')

    # Logistic regression
    explorer.predict_reorder()

class DataExplorer:
    def __init__(self, df, sample_size=None):
        if sample_size:
            self.df = df.sample(n=sample_size, random_state=42)
            print(f"Using a sample of {sample_size} rows for exploration.")
        else:
            self.df = df

    # Basic information
    def overview(self):
        print("Dataset Overview")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns & Types:")
        print(self.df.dtypes)
        print("\nFirst 10 rows:")
        print(self.df.head(10))

    # Missing values
    def missing_values(self):
        print("Missing Values")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])

    # Descriptive statistics
    def descript_stats(self):
        print("Descriptive Statistics")
        print(self.df.describe())

    # Histograms
    def plot_histograms(self):
        numeric = self.df.select_dtypes(include=['int64', 'float64'])
        if numeric.empty:
            print("No numeric columns to plot histograms.")
            return
        numeric.hist(bins=30, figsize=(12, 8))
        plt.tight_layout()
        plt.show()

    # BoxPlots
    def plot_boxplot(self):
        numeric = self.df.select_dtypes(include=['int64', 'float64'])
        for col in numeric.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

    # Correlation matrix
    def correlation_matrix(self):
        numeric = self.df.select_dtypes(include=['int64', 'float64'])
        if numeric.empty:
            print("No numeric columns for correlation.")
            return
        corr = numeric.corr()
        print("\n=== Correlation Matrix ===")
        print(corr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.show()

    # Linear regression example
    def linear_regression(self, x_col, y_col):
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Columns {x_col} or {y_col} not found.")
            return
        X = self.df[[x_col]].values.reshape(-1, 1)
        y = self.df[y_col].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X, y_pred, color='red', label='Regression')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Linear Regression: {x_col} vs {y_col}')
        plt.legend()
        plt.show()

        print(f"\nCoefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")

    # Pivot table - total items per user/product
    def pivot_table_summary(self, index_col,values_col='product_id', aggfunc='count'):
        if index_col not in self.df.columns:
            print(f"Columns {index_col} not found in dataframe.")
            return None
        pivot = pd.pivot_table(
            self.df,
            index=index_col,
            values=values_col,
            aggfunc=aggfunc,
        ).sort_values(by=values_col, ascending=False)
        print(f"\nPivot Table: {aggfunc} of {values_col} by {index_col}")
        print(pivot.head(50))
        return pivot

    # Reorder probability per product
    def reorder_probability(self):
        if 'reordered' not in self.df.columns or 'product_id' not in self.df.columns:
            print("Required columns not found for reorder probability.")
            return
        prob = self.df.groupby('product_id')['reordered'].mean().sort_values(ascending=False)
        print("\nTop 10 Products by Reorder Probability:")
        print(prob.head(10))

        # Plot top 20
        plt.figure(figsize=(12, 6))
        prob.head(20).plot(kind='bar', color='skyblue')
        plt.ylabel("Reorder Probability")
        plt.xlabel("Product ID")
        plt.title("Top 20 Products by Reorder Probability")
        plt.show()

        return prob

    def predict_reorder(self, feature_cols=None):
        if 'reordered' not in self.df.columns:
            print("Column 'reordered' not found.")
            return

        # Default features if none provided
        if feature_cols is None:
            feature_cols = ['total_items_in_order', 'total_orders_by_user', 'add_to_cart_order']

        # Drop rows with missing features
        df_model = self.df.dropna(subset=feature_cols + ['reordered'])

        X = df_model[feature_cols]
        y = df_model['reordered']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        print("\nClassification Report")
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, y_prob))

        # Optional confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


