import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.classifier import HotelReviewClassifier, CATEGORIES
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics


MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].fillna("").tolist()
        self.sentiments = df["sentiment_label"].tolist()
        self.priorities = df["priority_label"].tolist()
        self.ratings = df["rating_normalized"].tolist()
        self.categories = df[[f"cat_{c}" for c in CATEGORIES]].values.astype(float)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "priority": torch.tensor(self.priorities[idx], dtype=torch.long),
            "rating": torch.tensor(self.ratings[idx], dtype=torch.float),
            "category": torch.tensor(self.categories[idx], dtype=torch.float),
        }


class MultiTaskLoss(nn.Module):
    def __init__(self, w_sentiment=1.0, w_category=0.5, w_rating=0.5, w_priority=0.5):
        super().__init__()
        self.w_sentiment = w_sentiment
        self.w_category = w_category
        self.w_rating = w_rating
        self.w_priority = w_priority
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs, batch):
        loss_s = self.ce(outputs["sentiment"], batch["sentiment"])
        loss_c = self.bce(outputs["category"], batch["category"])
        loss_r = self.mse(outputs["rating"], batch["rating"])
        loss_p = self.ce(outputs["priority"], batch["priority"])
        return (
            self.w_sentiment * loss_s
            + self.w_category * loss_c
            + self.w_rating * loss_r
            + self.w_priority * loss_p
        )


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "xlm-roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ReviewDataset(train_df, tokenizer, max_length)
    val_ds = ReviewDataset(val_df, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HotelReviewClassifier(model_name).to(device)
    criterion = MultiTaskLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | train_loss={np.mean(train_losses):.4f} | val_loss={val_loss:.4f}")
        print(f"  sentiment_f1={val_metrics['sentiment_f1']:.4f} | rating_mae={val_metrics['rating_mae']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("  -> Melhor modelo salvo.")

    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_sent_preds, all_sent_true = [], []
    all_rating_preds, all_rating_true = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch)
        losses.append(loss.item())

        all_sent_preds.extend(outputs["sentiment"].argmax(dim=-1).cpu().tolist())
        all_sent_true.extend(batch["sentiment"].cpu().tolist())
        all_rating_preds.extend((outputs["rating"] * 10).clamp(0, 10).cpu().tolist())
        all_rating_true.extend((batch["rating"] * 10).cpu().tolist())

    metrics = {
        **compute_classification_metrics(all_sent_true, all_sent_preds, prefix="sentiment"),
        **compute_regression_metrics(all_rating_true, all_rating_preds, prefix="rating"),
    }
    return np.mean(losses), metrics
