import argparse, os, joblib, optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.data_loader import load_csv
from src.features import build_preprocessor
from src.models import model_candidates

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--out", default="models/best.joblib")
args = ap.parse_args()

X, y = load_csv(args.data, args.target)
pre = build_preprocessor(X)

def objective(trial):
    model = model_candidates(trial)
    pipe = Pipeline([("pre", pre), ("model", model)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_model = model_candidates(study.best_trial)
pipe = Pipeline([("pre", pre), ("model", best_model)])
pipe.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, args.out)
print("Saved", args.out, "best_trial", study.best_trial.params)
