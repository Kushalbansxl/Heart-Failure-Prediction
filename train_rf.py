import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from constants import *

def main():
    print("Loading tabular data...")
    csv_data = pd.read_csv(CSV_PATH)
    X_csv = csv_data.drop("DEATH_EVENT", axis=1)
    y_csv = csv_data["DEATH_EVENT"]

    X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
        X_csv, y_csv, test_size=0.2, random_state=42
    )
    print("Tabular data loaded and split successfully.")

    print("Starting RandomForest model training with hyperparameter tuning...")
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, rf_params, cv=3, scoring="accuracy", n_jobs=1)
    grid_search.fit(X_train_csv, y_train_csv)
    print("RandomForest model training completed.")

    best_rf_model = grid_search.best_estimator_
    print(f"Best RandomForest parameters: {grid_search.best_params_}")

    print("Evaluating RandomForest model...")
    y_pred_csv = best_rf_model.predict(X_test_csv)
    rf_scores = {
        "accuracy": accuracy_score(y_test_csv, y_pred_csv),
        "f1_score": f1_score(y_test_csv, y_pred_csv),
        "classification_report": classification_report(
            y_test_csv, y_pred_csv, output_dict=True
        ),
    }
    print(f"RandomForest Evaluation Scores: {rf_scores}")

    print("Saving RandomForest model and evaluation scores...")
    with open(RF_SCORE, "w") as rf_score_file:
        json.dump(rf_scores, rf_score_file)
    joblib.dump(best_rf_model, RF_MODEL)
    print("RandomForest model and scores saved successfully.")

if __name__ == "__main__":
    main()
