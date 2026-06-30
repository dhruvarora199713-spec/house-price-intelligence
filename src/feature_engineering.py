# ============================================
# Step 40: Load Outlier-Handled Dataset
# ============================================

# Purpose:
# Load dataset after
# outlier handling.

import pandas as pd 
import numpy as np 
from load_data import load_processed_data
df = load_processed_data("data/processed/outliers_handled.csv")

print("\nDataset Shape")
print(df.shape)


# ============================================
# Step 41: Analyze Features for Engineering
# ============================================

# Purpose:

# Organize features into
# logical groups before
# creating new features.

feature_groups = {
    "TimeFeatures" : [
        
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MoSold",
        "YrSold"
        
    ],
    
    "Area Features" : [
        
        "LotFrontage",
        "LotArea",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea"

    ],

    "Bathroom Features": [
        
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath"

    ],

    "Room Features": [

        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageCars"

    ],

    "Quality Features": [

        "OverallQual",
        "OverallCond",
        "ExterQual",
        "ExterCond",
        "KitchenQual",
        "HeatingQC",
        "BsmtQual",
        "BsmtCond",
        "GarageQual",
        "GarageCond",
        "FireplaceQu",
        "PoolQC"

    ]

}

print("\nFeature Groups:\n")

for group, columns in feature_groups.items():

    print("\n" + "=" * 50)
    print(group)
    print("=" * 50)
    for col in columns:
        print(col)


# ============================================
# Step 42: Create Engineered Features
# ============================================

# Purpose:
# Create meaningful features
# from existing attributes.

# House Age
df["HouseAge"] = (
    
    df["YrSold"] - df["YearBuilt"]
)

# Year since last remodeled
df["RemodelAge"] = (
    
    df["YrSold"] - df["YearRemodAdd"]
).clip(lower=0)

# Has garage or not 
df["HasGarage"] = (
    
    df["GarageArea"] > 0
).astype(int)

#Garage Age
df["GarageAge"] = (
    
    df["YrSold"] - df["GarageYrBlt"]
)
df.loc[

    df["HasGarage"] == 0,
    "GarageAge"
    
] = 0

#Total Bathrooms 
df["TotalBathrooms"] = (
    
    df["FullBath"] 
    + (0.5 * df["HalfBath"])
    + df["BsmtFullBath"]
    + (0.5 * df["BsmtHalfBath"])
    
)

# Total Porch Area
df["TotalPorchArea"] = (

    df["OpenPorchSF"]
    + df["EnclosedPorch"]
    + df["3SsnPorch"]
    + df["ScreenPorch"]

)

# whether the house is remodeled or not 
df["IsRemodeled"] = (
    
    df["YearBuilt"] != df["YearRemodAdd"]
    
).astype(int)

# Basement Availability
df["HasBasement"] = (
    
    df["TotalBsmtSF"] > 0

).astype(int)


# Pool Availability
df["HasPool"] = (

    df["PoolArea"] > 0

).astype(int)

# Fireplace Availability
df["HasFireplace"] = (

    df["Fireplaces"] > 0

).astype(int)

print("\nEngineered features created successfully.")



# ============================================
# Step 43: Verify Engineered Features
# ============================================

# Purpose:
# Verify newly created
# engineered features.

engineered_features = [

    "HouseAge",
    "RemodelAge",
    "GarageAge",
    "TotalBathrooms",
    "TotalPorchArea",
    "IsRemodeled",
    "HasGarage",
    "HasBasement",
    "HasPool",
    "HasFireplace"

]

print("\nEngineered Feature Summary:\n")

summary = (
    df[engineered_features]
    .describe(include="all")
)

print(summary)

print("\nMissing Values:\n")

print(
    df[engineered_features]
    .isnull()
    .sum()
)


# ============================================
# Step 44: Analyze Feature Skewness
# ============================================

# Purpose:
# Measure skewness of
# numerical features.

numerical_columns = (
    df.select_dtypes(
        include=["int64" , "float64"]
    ).columns
)

skewness = (
    df[numerical_columns]
    .skew()
    .sort_values(ascending=False
    )
)

print("\nFeature Skewness:\n")
print(skewness)



# ============================================
# Step 45: Identify Features for
# Log Transformation
# ============================================

# Purpose:
# Select highly skewed
# continuous numerical
# features suitable for
# log transformation.

excluded_features = [
    "Id",
    "SalePrice",
    "MSSubClass",
    "MoSold",
    "YrSold",
    "MiscVal",
    "PoolArea",
    
    "HasGarage",
    "HasBasement",
    "HasPool",
    "HasFireplace",
    "IsRemodeled",
    
    "KitchenAbvGr",
    "BsmtHalfBath"
    
]

skewness_threshold = 1.0

log_transform_features = (
    
    skewness [
        skewness.abs() > skewness_threshold
    ]
    .drop(
        labels = excluded_features ,
        errors = "ignore"
    )
    .index
    .tolist()
    
)

print("\nFeatures Selected For Log Transformation:\n")

for feature in log_transform_features:
    print(feature)
    
    
# ============================================
# Step 46: Apply Log Transformation
# ============================================

# Purpose:
# Apply log1p transformation
# to highly skewed continuous
# numerical features.

selected_log_features = [

    "3SsnPorch",
    "LowQualFinSF",
    "BsmtFinSF2",
    "ScreenPorch",
    "EnclosedPorch",
    "MasVnrArea",
    "LotFrontage",
    "OpenPorchSF",
    "TotalPorchArea",
    "BsmtFinSF1",
    "WoodDeckSF"

]


for col in selected_log_features:
    df[col]= np.log1p(df[col])
print("\nLog transformation completed successfully.")


# ============================================
# Step 47: Verify Log Transformation
# ============================================

# Purpose:
# Verify reduction in
# skewness after applying
# log transformation.

updated_skewness = (

    df[log_transform_features]
    .skew()
    .sort_values(
        ascending=False
    )

)

print("\nSkewness After Log Transformation:\n")
print(updated_skewness)


# ============================================
# Step 48: Analyze Categorical Features
# ============================================

# Purpose:
# Identify categorical
# features for encoding.

categorical_columns = (
    df.select_dtypes(
        include=["object" , "str"]
    ).columns
)
print("\nCategorical Features:\n")

for col in categorical_columns :
    print(col)
    
print("\nTotal Categorical Features:")
print(len(categorical_columns))



# ============================================
# Step 49: Classify Categorical Features
# ============================================

# Purpose:
# Separate categorical
# features into ordinal
# and nominal groups.

ordinal_features = [

    "LotShape",
    "LandSlope",

    "ExterQual",
    "ExterCond",

    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",

    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",

    "GarageFinish",
    "GarageQual",
    "GarageCond",

    "PoolQC",
    "Fence",
    
    "Utilities",
    "CentralAir",
    "PavedDrive"

]

nominal_features = [

    col
    for col in categorical_columns
    if col not in ordinal_features

]

print("\nOrdinal Features:\n")

for feature in ordinal_features:
    print(feature)


print("\nTotal Ordinal Features:")
print(len(ordinal_features))
print("\nNominal Features:\n")

for feature in nominal_features:
    print(feature)

print("\nTotal Nominal Features:")
print(len(nominal_features))



# ============================================
# Step 50: Define Ordinal Encoding Mappings
# ============================================

# Purpose:
# Define numerical mappings
# for ordinal categorical
# features.

ordinal_mappings = {

    "LotShape": {
        "Reg": 0,
        "IR1": 1,
        "IR2": 2,
        "IR3": 3
    },

    "LandSlope": {
        "Sev": 0,
        "Mod": 1,
        "Gtl": 2
    },

    "ExterQual": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "ExterCond": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "BsmtQual": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "BsmtCond": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "BsmtExposure": {
        "None": 0,
        "No": 1,
        "Mn": 2,
        "Av": 3,
        "Gd": 4
    },

    "BsmtFinType1": {
        "None": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6
    },

    "BsmtFinType2": {
        "None": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6
    },

    "HeatingQC": {
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "KitchenQual": {
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "FireplaceQu": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "GarageFinish": {
        "None": 0,
        "Unf": 1,
        "RFn": 2,
        "Fin": 3
    },

    "GarageQual": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "GarageCond": {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    },

    "PoolQC": {
        "None": 0,
        "Fa": 1,
        "TA": 2,
        "Gd": 3,
        "Ex": 4
    },

    "Fence": {
        "None": 0,
        "MnWw": 1,
        "GdWo": 2,
        "MnPrv": 3,
        "GdPrv": 4
    },

    "Utilities": {
        "ELO": 0,
        "NoSeWa": 1,
        "NoSewr": 2,
        "AllPub": 3
    },

    "CentralAir": {
        "N": 0,
        "Y": 1
    },

    "PavedDrive": {
        "N": 0,
        "P": 1,
        "Y": 2
    }

}

print("\nOrdinal mappings created successfully.")


# ============================================
# Step 51: Apply Ordinal Encoding
# ============================================

# Purpose:
# Apply ordinal encoding
# using predefined mappings.

for feature , mapping in ordinal_mappings.items():
    df[feature] = (
        df[feature]
        .map(mapping)
    )
    
print("\nOrdinal encoding completed successfully.")


# ============================================
# Step 52: Apply One-Hot Encoding
# ============================================

# Purpose:
# Convert nominal categorical
# features into binary
# indicator variables.

df = (
    pd.get_dummies(
        df,
        columns=nominal_features,
        dtype = int
    )
)
print("\nOne-Hot Encoding completed successfully.")



# ============================================
# Step 53: Final Dataset Verification
# ============================================

# Purpose:
# Verify dataset after
# feature engineering
# and encoding.

print("\nFinal Dataset Shape:\n")
print(df.shape)


print("\nRemaining Missing Values:\n")

print(
    df.isnull()
    .sum()
    .sum()
)


print("\nRemaining Object Columns:\n")

object_columns = (

    df.select_dtypes(
        include=["object" , "str"]
    )
    .columns
)

print(object_columns)


print("\nDuplicate Rows:\n")

print(
    df.duplicated()
    .sum()
)


print("\nData Types:\n")

print(
    df.dtypes
    .value_counts()
)


# ============================================
# Step 55: Display Final Dataset Summary and Save Feature-Engineered Dataset
# ============================================

print("\nFinal Dataset Summary")
print("=" * 50)

print(f"Rows              : {df.shape[0]}")
print(f"Columns           : {df.shape[1]}")
print(f"Numerical Columns : {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
print(f"Missing Values    : {df.isnull().sum().sum()}")
print(f"Duplicate Rows    : {df.duplicated().sum()}")

df.to_csv("data/processed/feature_engineered.csv",index=False)

print("\nFeature-engineered dataset saved successfully.")

