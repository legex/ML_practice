import os
import mlflow
from urllib.request import pathname2url
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from utils.datapreprocessing import DataProcessing
import pandas as pd

df = pd.read_csv("ML_practice\Student_study\student_habits_performance.csv")
dp = DataProcessing(df, target_col='exam_score')
X_train, X_test, y_train, y_test = dp.split()


lassoreg = Lasso()
params = {
            "alpha": [0.1, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002]
        }
grid = GridSearchCV(lassoreg, params, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid.fit(X_train, y_train)
best_models = grid.best_estimator_
best_params = grid.best_params_

artifact_path = os.path.abspath("mlruns")
#tracking_uri = "file://" + pathname2url(artifact_path)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("student_habits_performance")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("mse", mean_squared_error(y_test, best_models.predict(X_test)))
    signature = infer_signature(X_train, best_models.predict(X_train))
    mlflow.sklearn.log_model(
        best_models,
        artifact_path="student_habits_performance",
        signature=signature,
        registered_model_name="student_habits_performance"
    )
