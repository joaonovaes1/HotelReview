import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_raw(filename: str = "hotel_reviews.csv") -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    return df


def load_processed(filename: str = "reviews_labeled.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    return pd.read_csv(path, low_memory=False)


def save_processed(df: pd.DataFrame, filename: str = "reviews_labeled.csv") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)


def split_dataset(df: pd.DataFrame, seed: int = 42):
    train, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["sentiment_label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp["sentiment_label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Formato não suportado: {name}. Use CSV ou Excel.")
