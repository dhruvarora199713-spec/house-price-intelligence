# ============================================
# Helper Function
# ============================================

# Purpose:
# Calculate IQR lower and upper
# bounds for a numerical feature.

def calculate_iqr_bounds(df, column):

    Q1 = (

        df[column]

        .quantile(0.25)

    )

    Q3 = (

        df[column]

        .quantile(0.75)

    )

    IQR = (

        Q3 - Q1

    )

    lower_bound = (

        Q1 - 1.5 * IQR

    )

    upper_bound = (

        Q3 + 1.5 * IQR

    )

    return (
        Q1,
        Q3,
        IQR,
        lower_bound,
        upper_bound

    )




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

outlier_candidate_features = [
    "SalePrice",
    "LotArea",
    "GrLivArea",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF"
]

print("\nCandidate Columns For Outlier Analysis:\n")

for col in outlier_candidate_features:
    print(col)
    
    
# ============================================
# Step 33: Generate Boxplots
# ============================================

# Purpose:
# Visualize potential outliers
# in important numerical features.

import matplotlib.pyplot as plt 
import seaborn as sns 

for col in outlier_candidate_features:
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

for col in outlier_candidate_features:

    Q1, Q3, IQR, lower_bound, upper_bound = ( calculate_iqr_bounds(df, col))

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

for col in outlier_candidate_features :

    Q1, Q3, IQR, lower_bound, upper_bound = (
        calculate_iqr_bounds(df,col)
        )

    outliers = df[

        (df[col] < lower_bound)

        |

        (df[col] > upper_bound)

    ]

    print("\n" + "=" * 50)

    print(col)

    print("=" * 50)

    print(f"Number of Outliers: {len(outliers)}")

    print(f"Percentage: {(len(outliers)/len(df))*100:.2f}%")
    
# ============================================
# Step 36: Select Outlier Treatment Strategy
# ============================================

# Purpose:
# Define the outlier handling
# strategy for each feature.

outlier_treatment = {

    "SalePrice": "Keep",

    "LotArea": "Cap",

    "GrLivArea": "Cap",

    "GarageArea": "Cap",

    "TotalBsmtSF": "Cap",

    "1stFlrSF": "Cap"

}

print("\nSelected Outlier Treatment Strategy:\n")

for feature , strategy in outlier_treatment.items() :
    print(f"{feature} --> {strategy}")
    
# ============================================
# Step 37: Apply IQR-Based Outlier Capping
# ============================================

# Purpose:
# Cap outlier values using
# the IQR lower and upper bounds.

features_to_cap =[
    "LotArea",
    "GrLivArea",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF"
]

for col in features_to_cap:
    Q1,Q2,IQR,lower_bound,upper_bound =(calculate_iqr_bounds(df,col))
    df[col]=(
        df[col]
        .clip(
            lower=lower_bound,
            upper=upper_bound
        )
     )
print("\nOutlier capping completed successfully.")


# ============================================
# Step 38: Verify Outlier Reduction
# ============================================

# Purpose:
# Verify that outliers have been
# successfully capped.

for col in features_to_cap :
    Q1,Q2,IQR, lower_bound,upper_bound = (calculate_iqr_bounds(df,col))
    outliers = df[
        (df[col] < lower_bound)
    
        |
    
        (df[col] > upper_bound)]
    print("\n" + "=" * 50)

    print(col)

    print("=" * 50)

    print(f"Remaining Outliers: {len(outliers)}")
    
    
# ============================================
# Step 39: Save Outlier-Handled Dataset
# ============================================

# Purpose:
# Save dataset after
# outlier treatment.

df.to_csv( "data/processed/outliers_handled.csv",index=False)

print("\nDataset saved successfully.")