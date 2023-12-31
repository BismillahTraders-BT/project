import pandas as pd
from mlflow import MlflowClient
import mlflow.sklearn, mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

client = MlflowClient(tracking_uri="http://localhost:8080")

random_forest_experiment = mlflow.set_experiment("Random Forest")
run_name = "Random Forest Run 1"
artifact_path = "rf_model"

data = pd.read_csv('data/processed_data.csv')

X = data.drop(['Timestamp', 'Reading'], axis=1)
y = data['Reading']

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=30, max_depth=5)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")


params = {
    "n_estimators": 100, 
    "random_state": 42, 
    "max_depth": 5
}


metrics = {
    "mse": mse,
    "mae": mae,
    "r2": r2,
    "rmse": rmse
}

# Initiate a run, setting the `run_name` parameter
with mlflow.start_run(run_name=run_name) as run:
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    # Log an instance of the trained model for later use
    mlflow.sklearn.log_model(sk_model=rf, input_example=X_test, artifact_path=artifact_path)