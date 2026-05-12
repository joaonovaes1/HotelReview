import pandas as pd

EXPECTED_COLS = {
    "offering_id", "user_id", "overall",
    "value", "service", "location", "rooms",
    "cleanliness", "sleep_quality", "review_text",
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    return df
