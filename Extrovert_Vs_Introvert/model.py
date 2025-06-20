import os
import pandas as pd
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from utils.datapreprocessing import DataProcessing
from joblib import dump

df = pd.read_csv("ML_practice\Extrovert_Vs_Introvert\personality_datasert.csv")
dp = DataProcessing(df,target_col='Personality')
X_train,X_test,y_train,y_test = dp.split()

models = {
    # "SGD": {
    #     "model": SGDClassifier(),
    #     "params": {
    #         "alpha": [0.1, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002],
    #         "loss": ['hinge', 'modified_huber', 'log_loss'],
    #         "max_iter": [300, 500, 800, 1000, 1500]
    #     }
    # },
    "Ridge": {
        "model": RidgeClassifier(),
        "params": {
            "alpha": [0.1, 0.5, 0.004, 0.001, 0.005, 0.0006, 0.002],
            "fit_intercept": [True, False],
            "max_iter": [1, 3, 5, 7, 10, 100, 200, 250]
        }
    },
    # "KNN": {
    #     "model": KNeighborsClassifier(),
    #     "params": {
    #         "n_neighbors": [5, 10, 20, 15, 30, 35, 50, 40, 45],
    #         "algorithm": ["ball_tree", "kd_tree", "brute"],
    #         "leaf_size": [55, 10, 20, 15, 30, 35, 50, 40, 45]
    #     }
    # },
    "XGBoost": {
        "model": XGBClassifier(alpha=0.1, verbosity=1),
        "params": {
            "n_estimators": [150, 200, 250],
            "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.001, 0.005],
            "max_depth": [3, 5, 7],
            "subsample": [0.5, 0.7, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "min_samples_split": [1, 2, 3, 4, 5, 8, 9, 6, 10],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 20, 15, 3, 35, 4, 25],
            "max_leaf_nodes": [None, 5, 10, 20, 15, 3, 35, 4, 25],
            "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
    }
}

best_models = {}
best_score = 0
best_model_name = ""
best_model_obj = None

for name, mp in models.items():
    print(f"\nRunning GridSearchCV for: {name}")
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {accuracy}")
    print(f"precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"confusion: {confusion_matrix(y_test, y_pred)}")
    print(f"f1_score: {f1_score(y_test, y_pred, average='weighted')}")
    best_models[name] = grid.best_estimator_

    if accuracy > best_score:
        best_score = accuracy
        best_model_name = name
        best_model_obj = grid.best_estimator_
        print(f"New best model found: {best_model_name} with accuracy {best_score}")

print(f"\nBest overall model: {best_model_name} with accuracy: {best_score}")
save_dir = os.getcwd()

# Save model
model_path = os.path.join(f"{save_dir}\ML_practice\Extrovert_Vs_Introvert", f"{best_model_name}_best_classification.joblib")
dump(best_model_obj, model_path)
