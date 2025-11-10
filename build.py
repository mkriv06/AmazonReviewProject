import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from feature_extraction import extract_features
from preprocess import preprocess_text

include_pos = True

csv_path = Path(__file__).resolve().parent / "fake-reviews.csv"
df = pd.read_csv(csv_path)
df["cleaned_text"] = df["text_"].apply(preprocess_text)
df = extract_features(df, include_pos)
processed_path = Path(__file__).resolve().parent / "processed-dataset.csv"
df.to_csv(processed_path, index=False)
