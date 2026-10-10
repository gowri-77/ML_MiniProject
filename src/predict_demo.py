from pathlib import Path
import joblib
from src.preprocess import load_data

ROOT = Path(__file__).resolve().parent.parent
NAMES = {1: "low", 2: "medium", 3: "destroyed"}

def main():
    model = joblib.load(ROOT / "models" / "lgbm.joblib")
    _, X_val, _, y_val = load_data()
    sample = X_val.sample(5, random_state=7)
    pred = model.predict(sample)

    print(f"{'building_id':>11} | {'true grade':>10} | {'predicted grade':>15} | status")
    print("-" * 55)
    for bid, true, p in zip(sample.index, y_val.loc[sample.index], pred):
        flag = "OK" if true == p else "WRONG"
        print(f"{bid:>11} | {NAMES[true]:>10} | {NAMES[p]:>15} | {flag}")

if __name__ == '__main__':
    main()
