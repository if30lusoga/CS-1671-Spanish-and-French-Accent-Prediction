import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def short(a):
    if a.startswith("España"):
        # Spain regions: cut off the parenthesized list of places
        return a.split("(")[0].strip()
    else:
        # everything else: keep the text before the colon
        return a.split(":")[0]

dev = pd.read_csv("data/dev_filtered.csv") 
test = pd.read_csv("data/test_filtered.csv")
X_tr = np.load("data/dev_features.npy")
X_te = np.load("data/test_features.npy")
y_tr = dev.accents.map(short) 
y_te = test.accents.map(short)

clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("accuracy", accuracy_score(y_te, pred))
print("macro-F1", f1_score(y_te, pred, average="macro"))
print(classification_report(y_te, pred, zero_division=0))
print("train accuracy", clf.score(X_tr, y_tr))