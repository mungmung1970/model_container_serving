import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ✅ 로컬 / Docker 공통 경로
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ART_DIR = Path(os.getenv("ART_DIR", "artifacts"))


FEATURE_COLUMNS = [
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


def main():
    train_path = DATA_DIR / "mission15_train_add.csv"
    test_path = DATA_DIR / "mission15_test.csv"

    print(f"[INFO] Train data: {train_path.resolve()}")
    print(f"[INFO] Artifact dir: {ART_DIR.resolve()}")

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    df = pd.read_csv(train_path)

    X = df[FEATURE_COLUMNS]
    y = df["Performance Index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # --- artifacts ---
    ART_DIR.mkdir(parents=True, exist_ok=True)

    (ART_DIR / "metrics.json").write_text(
        json.dumps({"rmse": rmse}, indent=2), encoding="utf-8"
    )

    (ART_DIR / "features.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )

    # --- ONNX ---
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(ART_DIR / "model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 추론용 test 제공 (Docker 기준)
    if test_path.exists():
        shutil.copy(test_path, ART_DIR / "mission15_test.csv")

    print(f"✅ Training complete. RMSE = {rmse:.4f}")


if __name__ == "__main__":
    main()
