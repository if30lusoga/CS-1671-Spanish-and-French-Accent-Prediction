import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def short(a):
    if a.startswith("España"):
        return a.split("(")[0].strip()
    else:
        return a.split(":")[0]

dev = pd.read_csv("data/dev_filtered.csv") 
test = pd.read_csv("data/test_filtered.csv")

y_tr = dev.accents.map(short)
y_te = test.accents.map(short)

P_tr = np.load("data/dev_mfcc.npy")
M_tr = np.load("data/dev_mfcc.npy")

P_te = np.load("data/test_praat.npy")
M_te = np.load("data/test_mfcc.npy")

feature_sets = {
    "praat only": (P_tr, P_te),
    "mfcc only": (M_tr, M_te),
    "praat + mfcc": (np.hstack([P_tr, M_tr]), np.hstack([P_te, M_te])),
}

for name, (X_tr, X_te) in feature_sets.items():
    clf = make_pipeline(SimpleImputer(), StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print("=====", name, f"({X_tr.shape[1]} features)")
    print("train accuracy", clf.score(X_tr, y_tr))
    print("test accuracy", accuracy_score(y_te, pred))
    print("macro-F1", f1_score(y_te, pred, average="macro"))
    print(classification_report(y_te, pred, zero_division=0))
    for i in range(0, 630, 210):
        print(test.path[i], "| true:", y_te[i], "| predicted:", pred[i])