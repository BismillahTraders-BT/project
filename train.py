import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


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
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    mlflow.log_params(best_params)

    mlflow.log_metric("best_mse", best_mse)

    # Log the best model
    #mlflow.sklearn.log_model(best_model, "best_model")
    # Log the best model
    #mlflow.sklearn.log_model(best_model, "model/best_model")
    # Save the best model using joblib
    
    pickle.dump(best_model, open('model.pkl', 'wb'))

    print("MLflow Run ID:", run.info.run_id)