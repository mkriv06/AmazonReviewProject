import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from pathlib import Path
from xgboost import plot_importance
from matplotlib import pyplot
import joblib
import json
import numpy as np


# Load trained model
model_dir = Path("./model")
baseline_model_path = model_dir / "review_classifier.pkl"
model = joblib.load(baseline_model_path)

csv_path = Path(__file__).resolve().parent / "processed-dataset.csv"
df = pd.read_csv(csv_path)

X = pd.get_dummies(
    df.drop(columns=["label", "text_", "cleaned_text"]), 
    columns=["category"]
)

label_map = {"OR": 1, "CG": 0}
y = df["label"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df)

plot_importance(model)
pyplot.show()


parameters = {
    "n_estimators": 500,
    "learning_rate": 0.3,
    "max_depth": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
}

thresh = 0.04  # Only keep features with importance >= 0.020
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)

model = XGBClassifier(
    best_params=parameters,
    n_estimators=100,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(select_X_train, y_train)

select_X_test = selection.transform(X_test)

y_pred = model.predict(select_X_test)
y_prob = model.predict_proba(select_X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_prob)
print(f"\nAUC Score: {auc_score:.4f}")

# Save the fine-tuned model
joblib.dump(model, model_dir / "review_classifier.pkl")
feature_names = X.columns.tolist()

# Save model metadata

model_metadata = {
    "best_params": parameters,
    "num_original_features": len(feature_names),
}

with open(model_dir / "selection_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

'''
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=parameters,
    scoring="roc_auc",
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

print("\n" + "=" * 50)
print("GRID SEARCH RESULTS")
print("=" * 50)
print(f"Best parameters: {best_params}")
print(f"Best cross-validation AUC score: {best_score:.4f}")
print("=" * 50)

best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBest Model Performance on Test Set:")
print("Classification Report:\n", classification_report(y_test, best_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred))
print(f"Test AUC Score: {roc_auc_score(y_test, best_prob):.4f}")

baseline_pred = model.predict(X_test)
baseline_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("BASELINE vs BEST MODEL COMPARISON")
print("=" * 50)
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_prob):.4f}")
print(f"Best Model AUC: {roc_auc_score(y_test, best_prob):.4f}")
'''
'''
Best parameters: {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 500, 'reg_alpha': 0.5, 'reg_lambda': 1.0}
Best cross-validation AUC score: 0.8557

Baseline AUC: 0.8379
Best Model AUC: 0.8460
'''