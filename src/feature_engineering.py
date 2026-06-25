# ============================================
# Step 40: Load Outlier-Handled Dataset
# ============================================

# Purpose:
# Load dataset after
# outlier handling.

from load_data import load_data
df = load_data("data/processed/outliers_handled.csv")

print("\nDataset Shape")
print(df.shape)


