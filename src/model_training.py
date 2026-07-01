# ============================================
# Imports
# ============================================

import joblib
import pandas as pd

from load_data import load_processed_data

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# ============================================
# Step 56: Load Feature-Engineered Dataset
# ============================================

# Purpose:
# Load dataset after
# feature engineering .

df=load_processed_data("data/processed/feature_engineered.csv")

print("\nDataset Shape")
print(df.shape)


# ============================================
# Step 57: Separate Features and Target
# ============================================

# Purpose:
# Separate input features (X)
# and target variable (y)
# for model training.

X = df.drop(
    columns="SalePrice"
)

y = df["SalePrice"]

print("\nFeatures Shape")
print(X.shape)

print("\nTarget Shape")
print(y.shape)



# ============================================
# Step 58: Split Dataset into
# Training and Testing Sets
# ============================================

# Purpose:
# Split the dataset into
# training and testing
# sets for model training
# and evaluation.

X_train , X_test , y_train , y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)

print("\nTraining Features Shape")
print(X_train.shape)

print("\nTesting Features Shape")
print(X_test.shape)

print("\nTraining Target Shape")
print(y_train.shape)

print("\nTesting Target Shape")
print(y_test.shape)


# ============================================
# Step 59: Train Linear Regression Model
# ============================================

# Purpose:
# To train the model with Linear Regression

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train , y_train)

print("\nLinear Regression model trained successfully.")


# ============================================
# Step 60: Evaluate Linear Regression Model
# ============================================

# Purpose:
# Evaluate the trained
# Linear Regression model
# using the testing dataset.

linear_regression_predictions = (

    linear_regression_model.predict(X_test)

)

linear_regression_mae = (

    mean_absolute_error(

        y_test,

        linear_regression_predictions

    )

)

linear_regression_rmse = (

    root_mean_squared_error(

        y_test,

        linear_regression_predictions

    )

)

linear_regression_r2 = (

    r2_score(

        y_test,

        linear_regression_predictions

    )

)

print("\nLinear Regression Performance")
print(f"MAE  : {linear_regression_mae:.4f}")
print(f"RMSE : {linear_regression_rmse:.4f}")
print(f"R²   : {linear_regression_r2:.4f}")


# ============================================
# Step 61: Train Decision Tree Regressor.
# ============================================

# Purpose:
# To train the model with Decision Tree Regressor

Decision_Tree_model = DecisionTreeRegressor(random_state=42)
Decision_Tree_model.fit(X_train,y_train)

print("\nDecision Tree Regressor model trained successfully.")


# =================================================
# Step 62: Evaluate Decision Tree Regressor Model
# =================================================

# Purpose:
# Evaluate the trained
# Decision Tree Regressor model
# using the testing dataset.

Decision_Tree_predictions = Decision_Tree_model.predict(X_test)

Decision_Tree_mae = (
    mean_absolute_error(
        
        y_test,
        
        Decision_Tree_predictions
        
    )
)

Decision_Tree_rmse = (
    root_mean_squared_error(
        
          y_test,
          
          Decision_Tree_predictions
          
    )
)

Decision_Tree_r2 = (
    r2_score(
        
          y_test,
          
          Decision_Tree_predictions
    )
)


print("\nDecision Tree Regressor Performance")
print(f"MAE  : {Decision_Tree_mae:.4f}")
print(f"RMSE : {Decision_Tree_rmse:.4f}")
print(f"R²   : {Decision_Tree_r2:.4f}")


# ============================================
# Step 63: Train Random Forest model .
# ============================================

# Purpose:
# To train the model with Random Forest.

Random_Forest_model = RandomForestRegressor(n_estimators=100,random_state=42)
Random_Forest_model.fit(X_train , y_train)

print("\nRandom Forest model trained successfully.")



# =================================================
# Step 64: Evaluate Random Forest Model
# =================================================

# Purpose:
# Evaluate the trained
# Random Forest model
# using the testing dataset.


Random_Forest_model_predictions = Random_Forest_model.predict(X_test)

Random_Forest_mae = (
    mean_absolute_error(
        
        y_test,
        
       Random_Forest_model_predictions
        
    )
)

Random_Forest_rmse = (
    root_mean_squared_error(
        
          y_test,
          
         Random_Forest_model_predictions
          
    )
)

Random_Forest_r2 = (
    r2_score(
        
          y_test,
          
          Random_Forest_model_predictions
    )
)


print("\nRandom Forest Performance")
print(f"MAE  : {Random_Forest_mae:.4f}")
print(f"RMSE : {Random_Forest_rmse:.4f}")
print(f"R²   : {Random_Forest_r2:.4f}")


# ============================================
# Step 65: Compare Model Performance
# ============================================

# Purpose:
# Compare the performance
# of all trained models
# using MAE, RMSE and
# R² Score.

comparison_df = pd.DataFrame({

    "Model": [

        "Linear Regression",
        "Decision Tree",
        "Random Forest"

    ],

    "MAE": [

        linear_regression_mae,
        Decision_Tree_mae,
        Random_Forest_mae

    ],

    "RMSE": [

        linear_regression_rmse,
        Decision_Tree_rmse,
        Random_Forest_rmse

    ],

    "R² Score": [

        linear_regression_r2,
        Decision_Tree_r2,
        Random_Forest_r2

    ]

})

comparison_df = comparison_df.sort_values(

    by=["R² Score", "RMSE", "MAE"],
    ascending=[False, True, True]

).reset_index(drop=True)

print("\nModel Performance Comparison\n")
print(comparison_df)

best_model_name = comparison_df.loc[0, "Model"]

print(f"\nBest Model : {best_model_name}")

print("\nSelection Criteria:")
print("1. Highest R² Score")
print("2. Lowest RMSE (if R² is similar)")
print("3. Lowest MAE (if RMSE is also similar)")


# ============================================
# Step 66: Save Best Model
# ============================================

# Purpose:
# Save the best-performing
# trained model for
# future predictions.

if best_model_name == "Linear Regression":

    best_model = linear_regression_model

elif best_model_name == "Decision Tree":

    best_model = Decision_Tree_model

else:

    best_model = Random_Forest_model


joblib.dump(

    best_model,

    "models/best_model.pkl"

)

print("\nBest model saved successfully.")



# ============================================
# Step 67: Model Training Summary
# ============================================

# Purpose:
# Display the final
# model training summary.

print("\nModel Training Summary")
print("=" * 50)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")
print(f"Total Features   : {X_train.shape[1]}")

print(f"\nBest Model       : {best_model_name}")

print("\nTraining pipeline completed successfully.")