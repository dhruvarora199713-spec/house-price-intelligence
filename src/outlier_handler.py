# ============================================
# Step 29: Load Cleaned Dataset
# ============================================

# Purpose:
# Load dataset after missing value handling.

from load_data import load_data 
df = load_data("data/processed/missing_values_handled.csv")

print("\nDataset Shape")
print(df.shape)

# ============================================
# Step 30: Identify Numerical Features
# ============================================

# Purpose:
# Find all numerical columns
# available in the dataset.

numerical_column = (
    df.select_dtypes(include=["int64","float64"]).columns
)

print("\nNumerical Columns:\n")
print(numerical_column)


# ============================================
# Step 31: Generate Summary Statistics
# ============================================

# Purpose:
# Understand the distribution of
# numerical columns before
# performing outlier detection.

summary_stats = (
    df[numerical_column].describe()
)

print("\nSummary Statistics:\n")
print(summary_stats)


# ============================================
# Step 32: Select Candidate Columns
# ============================================

# Purpose:
# Select columns where outliers
# are most likely to exist.

candidate_columns = [
    "SalePrice",
    "LotArea",
    "GrLivArea",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF"
]

print("\nCandidate Columns For Outlier Analysis:\n")

for col in candidate_columns:
    print(col)
    
    
# ============================================
# Step 33: Generate Boxplots
# ============================================

# Purpose:
# Visualize potential outliers
# in important numerical features.

import matplotlib.pyplot as plt 
import seaborn as sns 

for col in candidate_columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(
        x=df[col]
        )
    
    plt.title(
        f"Boxplot of {col}"
        )

    plt.tight_layout()

    plt.savefig(
        f"plots/{col}_boxplot.png"
        )

    plt.show()
    
    
# ============================================
# Step 34: Calculate IQR Bounds For SalePrice
# ============================================

# Purpose:
# Calculate lower and upper bounds
# used to detect SalePrice outliers.

for col in candidate_columns:
    
    Q1 = (
        df[col]
        .quantile(0.25)
    )
    
    Q3 = (
        df[col]
        .quantile(0.75)
    )
    
    IQR = (
        Q3-Q1
    )
    
    lower_bound = (
        Q1 - 1.5*IQR
    )
     
    upper_bound = (
        Q3 + 1.5*IQR
    )           
    
    print("\n" + "=" * 50)

    print(col)

    print("=" * 50)

    print(f"Q1: {Q1}")

    print(f"Q3: {Q3}")

    print(f"IQR: {IQR}")

    print(f"Lower Bound: {lower_bound}")

    print(f"Upper Bound: {upper_bound}")
    

# ============================================
# Step 35: Count Outliers
# ============================================

# Purpose:
# Count outliers for each
# candidate feature.

for col in candidate_columns :
    
    Q1 = (
        df[col]
        .quantile(0.25)
    )
    
    Q3 = (
        df[col]
        .quantile(0.75)
    )
    
    IQR = (
        Q3-Q1
    )
    
    lower_bound = (
        Q1 - 1.5*IQR
    )
     
    upper_bound = (
        Q3 + 1.5*IQR
    )           

    outliers = df[
        (df[col] < lower_bound)
        |
        (df[col] > upper_bound)
    ]
    
    print("\n" + "=" * 50)

    print(col)

    print("=" * 50)

    print(

        f"Number of Outliers: {len(outliers)}"

    )

    print(

        f"Percentage: {(len(outliers)/len(df))*100:.2f}%"

    )
    
# ============================================
# Step 36: Inspect Outlier Rows
# ============================================

# Purpose:
# View actual outlier values
# for each candidate feature.

