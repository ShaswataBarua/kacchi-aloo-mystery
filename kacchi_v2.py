import glob, os
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score

def find(n):
    return (glob.glob(f"/kaggle/input/**/{n}", recursive=True) or glob.glob(f"**/{n}", recursive=True))[0]
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))

# Bangla digits -> numbers (not needed for the model itself, kept for safety)
b2e = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
for df in (train, test):
    for c in ["guests", "aloo_count"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.translate(b2e), errors="coerce")

# Tutul's extra zeros: ratio far above anything real (~max 2.7) -> divide by 10
for df in (train, test):
    df["r"] = df["aloo_count"] / df["guests"]
    big = df["r"] > 6
    df.loc[big, "aloo_count"] /= 10
    df["r"] = df["aloo_count"] / df["guests"]

# Improvement: train ONLY on weddings where aloo was actually counted
tr = train.dropna(subset=["r"])
y = tr["went_back_for_seconds"].astype(int)

# Only aloo-per-guest matters; every other column added nothing in CV
cv = StratifiedKFold(5, shuffle=True, random_state=1)
for d in (2, 3):
    m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=0)
    print(f"tree depth {d} on aloo_per_guest: CV acc =", round(cross_val_score(m, tr[["r"]], y, cv=cv).mean(), 4))

model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=0).fit(tr[["r"]], y)
print(export_text(model, feature_names=["aloo_per_guest"]))
test["r"] = test["r"].fillna(tr["r"].median())
pd.DataFrame({"wedding_id": test["wedding_id"], "went_back_for_seconds": model.predict(test[["r"]])}) \
  .to_csv("/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission_v2.csv", index=False)
print("saved")
