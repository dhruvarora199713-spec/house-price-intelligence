# =============
# constants 
# =============

feature_absent_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu"
]

actual_missing_columns = [
    "LotFrontage",
    "GarageYrBlt",
    "MasVnrArea",
    "Electrical"
]


# ============================================
# Step 16: Analyze Missing Features
# ============================================

# Purpose:
# Find columns with missing values
# and calculate missing percentages

import pandas as pd 
# Load Dataset
df=pd.read_csv("data/raw/train.csv")

# Calculate misssing percentages 
missing_percentage=(
    df.isnull()
      .sum()
      / len(df)
      * 100
)

# Keep only columns with missing values 
missing_percentage=(
    missing_percentage[
        missing_percentage > 0
    ]
    .sort_values(ascending=False)
)

print(missing_percentage)


# ============================================
# Step 17: Investigate High Missing Columns
# ============================================

# Purpose:
# Examine highly missing columns and
# understand what values they contain.
# Shows which values are present in a column and how many times each value occurs. 
# Gd = good , Fa = fair , Ex = excellent ...


for col in feature_absent_columns :
    print("\n" + "=" * 50)
    print(f"columns: {col}")
    print("=" * 50)
    print(df[col].value_counts())
    
    
# ============================================
# Step 18: Create Missing Value Strategy
# ============================================

# Purpose:
# Classify columns based on
# how missing values should be handled.


print("\nColumns where missing means feature absent:")
print(feature_absent_columns)

print("\nColumns where missing means actual missing data:")
print(actual_missing_columns)


# ============================================
# Step 19: Inspect Actual Missing Data Columns
# ============================================

# Purpose:
# Understand numerical columns that
# require value imputation.

numerical_missing_columns = [
    "LotFrontage",
    "GarageYrBlt",
    "MasVnrArea"
]

for col in numerical_missing_columns :
    print("\n" + "=" * 50)
    print(f"column: {col}")
    print("=" * 50)
    print(df[col].describe())
    

# ============================================
# Step 20: Handle Feature-Absent Columns
# ============================================

# Purpose:
# Missing values in these columns
# mean the feature is absent.

for col in feature_absent_columns:
    df[col] = df[col].fillna("None")
    
print("\nFeature absent columns handled successfully.")


# ============================================
# Step 21: Verify Feature-Absent Columns
# ============================================

# Purpose:
# Ensure missing values are removed.

print("\nVerification Results: \n")

for col in feature_absent_columns :
    print(f"{col} -> {df[col].isnull().sum()} missing values")

# ============================================
# Step 22: Handle Numerical Missing Values
# ============================================

# Purpose:
# Fill missing numerical values using median.

for col in numerical_missing_columns :
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    
print("\nNumerical missing values handled successfully.")


# =============================================
# Step 23: Check Remaining Missing Values
# =============================================
def show_missing(df):
    
    missing = (

        df.isnull()
          .sum()
    )

    missing = (

        missing[missing > 0]
        .sort_values(ascending=False)
    )

    print(missing)
    
show_missing(df)


# ============================================
# Step 24: Investigate Basement Missing Pattern
# ============================================

basement_columns = [

    "BsmtExposure",
    "BsmtFinType2",
    "BsmtQual",
    "BsmtCond",
    "BsmtFinType1"

]

for col in basement_columns:

    print("\n" + "=" * 50)
    print(col)
    print("=" * 50)

    print(df[col].value_counts())

# ================================================
# Step 25: Handle Remaining Feature-Absent Columns
# ================================================

remaining_feature_absent_columns = [
    "MasVnrType",
    
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    
    "BsmtExposure",
    "BsmtFinType2",
    "BsmtQual",
    "BsmtCond",
    "BsmtFinType1"
]

for col in remaining_feature_absent_columns:
    df[col]=df[col].fillna("None")

print("\nRemaining feature-absent columns handled successfully.")


# ============================================
# Step 26: Handle Electrical Missing Value
# ============================================

# Purpose:
# Fill missing Electrical values
# using the most frequent category.

mode_value = df["Electrical"].mode()[0]
df["Electrical"] = df["Electrical"].fillna(mode_value)

print("\nElectrical missing values handled successfully.")


# ============================================
# Step 27: Final Missing Value Verification
# ============================================

# Purpose:
# Verify that all missing values
# have been handled successfully.

print("\nRemaining Missing Values:\n")
show_missing(df)


# ============================================
# Step 28: Save Cleaned Dataset
# ============================================

# Purpose:
# Save dataset after missing value treatment.

df.to_csv("data/processed/missing_values_handled.csv",index=False)

print("\nDataset saved successfully.")

