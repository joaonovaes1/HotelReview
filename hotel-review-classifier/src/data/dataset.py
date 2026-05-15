import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ReviewDataset(Dataset):

    def __init__(self, csv_path: str, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    SENTIMENT_MAP = {"negativo": 0, "neutro": 1, "positivo": 2}
    PRIORITY_MAP  = {"normal": 0, "alta": 1}
    CATEGORY_COLS = [
        "limpeza", "atendimento", "localização",
        "alimentação", "preço", "conforto", "wifi", "instalações",
    ]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            row["review_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_sentiment": torch.tensor(self.SENTIMENT_MAP[row["sentiment"]], dtype=torch.long),
            "label_category":  torch.tensor(row[self.CATEGORY_COLS].values.astype(float), dtype=torch.float),
            "label_rating":    torch.tensor((row["overall"] - 1) / 4, dtype=torch.float),
            "label_priority":  torch.tensor(self.PRIORITY_MAP[row["priority"]], dtype=torch.long),
        }
