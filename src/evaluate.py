from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def log_result(name, model, X_train, y_train, X_val, y_val, train_time,
               train_sample=20000):
    """Score a fitted model and append one row to results/results.csv.

    Train scores use a random 20k sample of the training set, because
    predicting all 208k rows is slow for KNN.
    Returns the validation predictions (for confusion matrices).
    """
    Xs = X_train.sample(n=min(train_sample, len(X_train)), random_state=42)
    ys = y_train.loc[Xs.index]

    pred_tr = model.predict(Xs)
    pred_va = model.predict(X_val)

    row = {
        "model": name,
        "train_micro_f1": round(f1_score(ys, pred_tr, average="micro"), 4),
        "val_micro_f1": round(f1_score(y_val, pred_va, average="micro"), 4),
        "val_macro_f1": round(f1_score(y_val, pred_va, average="macro"), 4),
        "train_time_sec": round(train_time, 1),
    }
    RESULTS.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS, mode="a",
                               header=not RESULTS.exists(), index=False)
    print(row)
    return pred_va