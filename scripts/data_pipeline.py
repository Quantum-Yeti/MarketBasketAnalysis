from scripts.data_cleaning import clean_data, save_clean_data
from scripts.data_exploration import run_exploration
from scripts.data_loading import load_data

# Step 1: Load the data
data = load_data()

# Step 2: Clean & Merge the data
cleaned_df = clean_data(data)

# Step 3: Save the clean data
save_clean_data(cleaned_df)

# Step 4: Data exploration
de = run_exploration()