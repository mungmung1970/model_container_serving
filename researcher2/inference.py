import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

from preprocess import preprocess_for_inference


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
    print(f"[INFO] DATA_DIR = {DATA_DIR.resolve()}")
    print(f"[INFO] ART_DIR  = {ART_DIR.resolve()}")

    model_path = ART_DIR / "model.onnx"
    feature_path = ART_DIR / "features.json"
    test_path = DATA_DIR / "mission15_test.csv"

    for p in [model_path, feature_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {p}")

    # 1. 데이터 로드
    df = pd.read_csv(test_path)

    # 2. 전처리
    df_proc = preprocess_for_inference(df)

    # 3. feature 선택
    X = df_proc[FEATURE_COLUMNS].astype(np.float32)

    # 4. 중간 산출물
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df_proc[FEATURE_COLUMNS].to_csv(ART_DIR / "mission15_test_add.csv", index=False)

    # 5. 추론
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: X.values})[0]

    # 6. 결과 저장
    df_proc["prediction"] = pred
    df_proc.to_csv(ART_DIR / "result.csv", index=False)

    print("✅ Inference complete")
    print(" - mission15_test_add.csv")
    print(" - result.csv 생성")


if __name__ == "__main__":
    main()
