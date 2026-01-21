import json
import numpy as np
import pandas as pd
import onnxruntime as ort

ART_DIR = "/artifacts"

# load artifacts
features = json.load(open(f"{ART_DIR}/features.json"))
df = pd.read_csv(f"{ART_DIR}/test.csv")

# preprocessing (same as training)
df["extra_yn"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
df["study_efficiency"] = df["Previous Scores"] / df["Hours Studied"]
df["practice_ratio"] = df["Sample Question Papers Practiced"] / df["Hours Studied"]
df["study_sleep_ratio"] = df["Hours Studied"] / df["Sleep Hours"]
df["conditioned_study"] = df["Sleep Hours"] * np.log1p(df["Hours Studied"])
df["focus_index"] = df["Hours Studied"] / (df["extra_yn"] + 1)
df["learning_engagement"] = (
    df["Hours Studied"] + df["Sample Question Papers Practiced"] - df["extra_yn"] * 0.5
)
df["balance_score"] = (df["Sleep Hours"] / (df["Hours Studied"] + 1)) * (
    1 + df["extra_yn"]
)

X = df[features].astype(np.float32)

# inference
sess = ort.InferenceSession(f"{ART_DIR}/model.onnx")
input_name = sess.get_inputs()[0].name
pred = sess.run(None, {input_name: X.values})[0]

df["prediction"] = pred
df.to_csv(f"{ART_DIR}/result.csv", index=False)

print("Inference complete → result.csv saved")
