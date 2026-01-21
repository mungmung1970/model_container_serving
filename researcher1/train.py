import json
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


DATA_DIR = "/app/data"
ART_DIR = "/artifacts"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["extra_yn"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    df["study_efficiency"] = df["Previous Scores"] / df["Hours Studied"]
    df["practice_ratio"] = df["Sample Question Papers Practiced"] / df["Hours Studied"]
    df["study_sleep_ratio"] = df["Hours Studied"] / df["Sleep Hours"]
    df["conditioned_study"] = df["Sleep Hours"] * np.log1p(df["Hours Studied"])
    df["focus_index"] = df["Hours Studied"] / (df["extra_yn"] + 1)
    df["learning_engagement"] = (
        df["Hours Studied"]
        + df["Sample Question Papers Practiced"]
        - df["extra_yn"] * 0.5
    )
    df["balance_score"] = (df["Sleep Hours"] / (df["Hours Studied"] + 1)) * (
        1 + df["extra_yn"]
    )

    return df


def main():
    df = pd.read_csv(f"{DATA_DIR}/mission15_train_add.csv")
    df = preprocess(df)

    feature_cols = [
        "Hours Studied",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        "extra_yn",
        "study_efficiency",
        "practice_ratio",
        "conditioned_study",
        "focus_index",
        "learning_engagement",
        "balance_score",
    ]

    X = df[feature_cols]
    y = df["Performance Index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # --- save metrics ---
    with open(f"{ART_DIR}/metrics.json", "w") as f:
        json.dump({"rmse": rmse}, f, indent=2)

    # --- save feature schema ---
    with open(f"{ART_DIR}/features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # --- convert to ONNX ---
    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(f"{ART_DIR}/model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # --- provide test.csv for researcher2 ---
    shutil.copy(f"{DATA_DIR}/test.csv", f"{ART_DIR}/test.csv")

    print(f"Training complete. RMSE = {rmse:.4f}")


if __name__ == "__main__":
    main()
