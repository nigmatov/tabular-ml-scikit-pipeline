from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def model_candidates(trial=None):
    if trial and trial.suggest_categorical("model", ["rf","lr"]) == "lr":
        C = trial.suggest_float("lr_C", 1e-3, 10.0, log=True)
        return LogisticRegression(max_iter=1000, C=C)
    else:
        n = trial.suggest_int("rf_n_estimators", 50, 300)
        md = trial.suggest_int("rf_max_depth", 2, 10)
        return RandomForestClassifier(n_estimators=n, max_depth=md, random_state=42)
