import os
import mlflow
from urllib.request import pathname2url
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from utils.datapreprocessing import DataProcessing
import pandas as pd

df = pd.read_csv("ML_practice/Extrovert_Vs_Introvert/personality_datasert.csv")
dp = DataProcessing(df, target_col='Personality')
X_train, X_test, y_train, y_test = dp.split()

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

xgbclassmodel = XGBClassifier(verbosity=1)
params = {
            "n_estimators": [150, 200, 250],
            "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.001, 0.005],
            "max_depth": [3, 5, 7],
            "subsample": [0.5, 0.7, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0]
        }
grid = GridSearchCV(xgbclassmodel, params, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_res, y_res)
best_models = grid.best_estimator_
best_params = grid.best_params_

artifact_path = os.path.abspath("mlruns")
#tracking_uri = "file://" + pathname2url(artifact_path)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Extrovert_Vs_Introvert")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, best_models.predict(X_test)))
    signature = infer_signature(X_res, best_models.predict(X_res))
    mlflow.xgboost.log_model(
        best_models,
        artifact_path="Extrovert_Vs_Introvert_Model",
        signature=signature,
        registered_model_name="Extrovert_Vs_Introvert_Model"
    )
