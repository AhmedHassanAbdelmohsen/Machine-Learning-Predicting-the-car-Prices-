1️⃣ Load Dataset – Reads car data.csv into a Pandas DataFrame.

2️⃣ Compute Correlation Matrix – Generates correlations between numerical columns and saves them to a CSV file.

3️⃣ Initialize PyCaret – Sets up the regression environment, automatically handling preprocessing like missing values and feature scaling.

4️⃣ Train Model – Uses compare_models() to find the best-performing regression model and fine-tunes it with create_model().

5️⃣ Make Predictions – Uses the trained model to predict selling prices on the dataset.

6️⃣ Save Predictions – Exports the predicted selling prices to a CSV file.

