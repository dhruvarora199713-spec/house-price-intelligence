# ============================================
# Imports
# ============================================

import joblib
import pandas as pd

from load_data import load_processed_data


# ============================================
# Step 69: Load Trained Model
# ============================================

model = joblib.load("models/best_model.pkl")
print("\nBest model loaded successfully.")

# ============================================
# Step 70: Load Multiple Houses
# ============================================

# Purpose:
# Load multiple sample
# houses from the
# feature-engineered dataset.


df = load_processed_data(
    "data/processed/feature_engineered.csv"
)

start_index = 0           # to start prediction
end_index = len(df)       # len(df) means the entire dataset.

sample_houses = df.drop(
    columns="SalePrice"
).iloc[start_index:end_index]

actual_prices = df.loc[
    start_index:end_index - 1,
    "SalePrice"
]

print("\nSample houses loaded successfully.")
print("\nSample Houses Shape")
print(sample_houses.shape)


# ============================================
# Step 71: Predict House Prices
# ============================================

# Purpose:
# Predict prices for
# multiple houses.

predicted_prices = model.predict(
    sample_houses
)

print("\nPrediction completed successfully.")


# ============================================
# Step 72: Compare Predictions
# ============================================

# Purpose:
# Compare actual and
# predicted prices.

print("\nPrediction Summary")
print("=" * 65)

for i in range(len(sample_houses)):

    house_number = start_index + i

    actual_price = actual_prices.iloc[i]

    predicted_price = predicted_prices[i]

    prediction_error = abs(
        actual_price - predicted_price
    )

    prediction_error_percentage = (
        prediction_error / actual_price
    ) * 100

    print(f"\nHouse Index : {house_number}")
    print(f"Actual Price      : {actual_price:.2f}")
    print(f"Predicted Price   : {predicted_price:.2f}")
    print(f"Prediction Error  : {prediction_error:.2f}")
    print(f"Error Percentage  : {prediction_error_percentage:.2f}%")

    print("-" * 65)
    
# ============================================
# Step 73: Save Prediction Results
# ============================================

# Purpose:
# Save prediction results
# for future analysis.

prediction_results = pd.DataFrame({

    "House Index": range(
        start_index,
        end_index
    ),

    "Actual Price": actual_prices.values,

    "Predicted Price": predicted_prices,

    "Prediction Error": abs(
        actual_prices.values - predicted_prices
    ),

    "Error Percentage": (

        abs(
            actual_prices.values - predicted_prices
        )

        / actual_prices.values

    ) * 100

})

prediction_results.to_csv( "reports/prediction_results.csv",index=False)

print("\nPrediction results saved successfully.")

# ============================================
# Step 74: Prediction Summary
# ============================================

# Purpose:
# Display the final
# prediction summary.

average_error = prediction_results[
    "Prediction Error"
].mean()

average_error_percentage = prediction_results[
    "Error Percentage"
].mean()

minimum_error = prediction_results[
    "Prediction Error"
].min()

maximum_error = prediction_results[
    "Prediction Error"
].max()

best_prediction = prediction_results.loc[
    prediction_results["Prediction Error"].idxmin()
]

worst_prediction = prediction_results.loc[
    prediction_results["Prediction Error"].idxmax()
]

print("\nPrediction Summary")
print("=" * 60)

print(f"Total Houses Tested      : {len(prediction_results)}")
print(f"Average Prediction Error : {average_error:.2f}")
print(f"Average Error Percentage : {average_error_percentage:.2f}%")
print(f"Minimum Prediction Error : {minimum_error:.2f}")
print(f"Maximum Prediction Error : {maximum_error:.2f}")

print("\nBest Prediction")
print("-" * 60)
print(f"House Index : {best_prediction['House Index']}")
print(f"Error       : {best_prediction['Prediction Error']:.2f}")
print(f"Error %     : {best_prediction['Error Percentage']:.2f}%")

print("\nWorst Prediction")
print("-" * 60)
print(f"House Index : {worst_prediction['House Index']}")
print(f"Error       : {worst_prediction['Prediction Error']:.2f}")
print(f"Error %     : {worst_prediction['Error Percentage']:.2f}%")

print("\nPrediction pipeline completed successfully.")