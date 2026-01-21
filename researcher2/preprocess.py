import numpy as np
import pandas as pd


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Yes / No → 0 / 1
    df["extra_yn"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    # 파생 지표
    df["study_efficiency"] = df["Previous Scores"] / df["Hours Studied"]
    df["practice_ratio"] = df["Sample Question Papers Practiced"] / df["Hours Studied"]
    df["study_sleep_ratio"] = df["Hours Studied"] / df["Sleep Hours"]
    df["conditioned_study"] = df["Sleep Hours"] * np.log1p(df["Hours Studied"])
    df["focus_index"] = df["Hours Studied"] / (df["extra_yn"] + 1)
    df["growth_potential"] = df["Hours Studied"] / df["Previous Scores"]
    df["effort_result_ratio"] = df["Previous Scores"] / (
        df["Hours Studied"] + df["Sample Question Papers Practiced"]
    )
    df["learning_engagement"] = (
        df["Hours Studied"]
        + df["Sample Question Papers Practiced"]
        - df["extra_yn"] * 0.5
    )
    df["balance_score"] = (df["Sleep Hours"] / (df["Hours Studied"] + 1)) * (
        1 + df["extra_yn"]
    )

    return df
