import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data from the uploaded CSV file
file_path = 'data/dummy_sensor_data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
##print(df.head())

# Convert 'Timestamp' to a numeric format (Unix timestamp)
df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype('int64') // 10**9  # Use // for integer division

# Separate features and target
X = df.drop('Reading', axis=1)
y = df['Reading']

# Define a column transformer that will one-hot encode categorical variables
# and scale numerical features
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Machine_ID', 'Sensor_ID']),
        ('scaler', StandardScaler(), ['Timestamp'])
    ],
    remainder='passthrough'
)

# Apply the transformations
X_transformed = column_transformer.fit_transform(X)

# Split the transformed data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
# X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
# X_train_df.to_csv('train_data.csv', index=False)



# Display the shapes of the training and validation sets
print(f"X_train shape: {X_train.shape}, X_val shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_test.shape}")



model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_test)

# Define a smaller parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

# Initialize Grid Search with fewer cross-validation folds and limited parallel jobs
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=2, verbose=2)

# Perform grid search
grid_search.fit(X_train, y_train)

# Start an MLflow run# After grid search is completed
with mlflow.start_run() as run:

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Evaluation Errors: MSE[{mse}], MAE[{mae}], R2[{r2}], RMSE[{rmse}]")
    mlflow_metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
    }

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    mlflow.log_params(best_params)
    best_mse = mean_squared_error(y_test, best_model.predict(X_test))
    mlflow.log_metric("best_mse", best_mse)

    # Log the best model
    #mlflow.sklearn.log_model(best_model, "best_model")
    # Log the best model
    #mlflow.sklearn.log_model(best_model, "model/best_model")
    # Save the best model using joblib
    
    pickle.dump(best_model, open('model.pkl', 'wb'))

    print("MLflow Run ID:", run.info.run_id)