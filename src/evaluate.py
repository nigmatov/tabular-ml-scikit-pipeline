import argparse, joblib, shap, pandas as pd, matplotlib.pyplot as plt, os
from src.data_loader import load_csv

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--target", required=True)
args = ap.parse_args()

pipe = joblib.load(args.ckpt)
X, y = load_csv(args.data, args.target)
pred = pipe.predict(X)
acc = (pred == y).mean()
print("Accuracy:", acc)

os.makedirs("reports", exist_ok=True)
try:
    # Fall back to kernel SHAP for simplicity
    import numpy as np
    background = pipe.named_steps["pre"].transform(X.iloc[:20])
    expl = shap.KernelExplainer(pipe.named_steps["model"].predict_proba, background)
    shap_vals = expl.shap_values(pipe.named_steps["pre"].transform(X.iloc[:5]))
    shap.summary_plot(shap_vals[1], background, show=False)
    plt.savefig("reports/shap_summary.png", bbox_inches="tight")
    print("Saved SHAP plot to reports/shap_summary.png")
except Exception as e:
    print("SHAP visualization skipped:", e)
