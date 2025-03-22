from pycaret.regression import *
import pandas as pd

# Define file path
file_path = r"C:\Users\Ahmed Hassan\Downloads\ml projects\car data.csv"

# Step 1️⃣: Load the dataset
data = pd.read_csv(file_path)

# Compute correlation matrix (only for numerical columns)
correlation_matrix = data.corr(numeric_only=True)

# Step 2️⃣: Save correlation matrix
correlation_output_path = r"C:\Users\Ahmed Hassan\Downloads\ml projects\correlation_matrix.csv"
correlation_matrix.to_csv(correlation_output_path, index=False)
print(f"Correlation matrix saved to: {correlation_output_path}")

# Step 3️⃣: Reload the original dataset
data = pd.read_csv(file_path)

# Step 4️⃣: Initialize PyCaret (auto handles preprocessing)
exp = setup(data=data,
            target='Selling_Price',
            session_id=123,
            verbose=False)  # No 'ignore_low_variance' here!

# Step 5️⃣: Train the best model automatically
best_model = compare_models()
final_model = create_model(best_model)

# Step 6️⃣: Make predictions
predictions = predict_model(final_model, data=data)

# Step 7️⃣: Save predictions
predictions_output_path = r"C:\Users\Ahmed Hassan\Downloads\ml projects\predicted_selling_price.csv"
predictions.to_csv(predictions_output_path, index=False)
print(f"Predictions saved to: {predictions_output_path}")
