import os
import pandas as pd
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from utils.datapreprocessing import DataProcessing
from joblib import dump

# Load the dataset
# Ensure the path is correct for your environment
df = pd.read_csv("ML_practice\Housing_dataset\housing_price_dataset.csv")
dp = DataProcessing(df,target_col='Price')
X_train,X_test,y_train,y_test = dp.split()

models = {
    "Linear": {
        "model": LinearRegression(),
        "params": {
            # LinearRegression has no 'alpha'; empty or remove this if no hyperparameters
            "fit_intercept": [True, False]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.1, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [0.1, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002]
        }
    }
}

best_models = {}
best_score = float("-inf")   # R² can be negative, so start with -∞
best_model_name = ""
best_model_obj = None

for name, mp in models.items():
    print(f"\nRunning GridSearchCV for: {name}")
    
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    
    print(f"Best CV R2 Score (grid.best_score_): {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")
    print("Test Set R2 Score: ", r2)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    best_models[name] = grid.best_estimator_

    # Track the overall best model based on test R²
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_model_obj = grid.best_estimator_
        print(f"New best model found: {best_model_name} with R2 Score: {best_score:.4f}")
# Save the best model
print(f"\nBest overall regression model: {best_model_name} with R2 score: {best_score:.4f}")
save_dir = os.getcwd()

# Save model
model_path = os.path.join(f"{save_dir}\ML_practice\Housing_dataset", f"{best_model_name}_best_regressor.joblib")
dump(best_model_obj, model_path)
