import os
import re
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


TEXT_COL = "text"
LABEL_COL = "sentiment"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_data(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File tidak ditemukan: {raw_path}")
    return pd.read_csv(raw_path)


def preprocess(df: pd.DataFrame):
    missing_cols = [c for c in [TEXT_COL, LABEL_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ada di dataset: {missing_cols}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL])

    df["clean_review"] = df[TEXT_COL].astype(str).apply(clean_text)

    le = LabelEncoder()
    df["sentiment_label"] = le.fit_transform(df[LABEL_COL])

    final_df = df[["clean_review", "sentiment_label"]].reset_index(drop=True)
    return final_df, le


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="spiderman_youtube_review_raw/spiderman_youtube_review_raw.csv",
        help="Path ke file CSV mentah",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessing/spiderman_youtube_review_preprocessing",
        help="Folder output untuk menyimpan hasil preprocessing",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Membaca data mentah dari: {args.raw_path}")
    df_raw = load_raw_data(args.raw_path)

    print("[INFO] Melakukan preprocessing...")
    final_df, label_encoder = preprocess(df_raw)

    output_csv = os.path.join(
        args.output_dir, "spiderman_youtube_review_preprocessed.csv"
    )
    final_df.to_csv(output_csv, index=False)

    le_path = os.path.join(args.output_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)

    print(f"[INFO] Hasil preprocessing disimpan di: {output_csv}")
    print(f"[INFO] LabelEncoder disimpan di: {le_path}")
    print(f"[INFO] Jumlah data akhir: {len(final_df)}")


if __name__ == "__main__":
    main()
