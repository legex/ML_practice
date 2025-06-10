import pandas as pd
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from datapreprocessing import Dataprocessing

df = pd.read_csv("student_habits_performance.csv")
dp = Dataprocessing(df,target_col='exam_score')
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
            "alpha": [0.1, 1.0, 10.0, 100.0, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [0.1, 1.0, 10.0, 100.0, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002]
        }
    }
}

best_models = {}

for name, mp in models.items():
    print(f"\nRunning GridSearchCV for: {name}")
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(f"Best Score: {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")
    best_models[name] = grid.best_estimator_
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print("R2_Score: ", r2_score(y_test, y_pred))
