import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, mean_absolute_error


class MultiTaskLoss(nn.Module):

    def __init__(self, task_weights: dict = None):
        super().__init__()

        # pesos por tarefa — quanto cada loss contribui para o total
        self.task_weights = task_weights or {
            "sentiment": 1.0,
            "category":  1.0,
            "rating":    1.0,
            "priority":  0.5,
        }

        # class weights para sentimento: dataset tem 71% positivo, 9% negativo
        self.register_buffer("sentiment_class_w", torch.tensor([2.0, 1.0, 0.4]))

        self.loss_sentiment = nn.CrossEntropyLoss(weight=self.sentiment_class_w)
        self.loss_category  = nn.BCEWithLogitsLoss()
        self.loss_rating    = nn.MSELoss()
        self.loss_priority  = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        l_sentiment = self.loss_sentiment(outputs["sentiment"], labels["label_sentiment"])
        l_category  = self.loss_category(outputs["category"],  labels["label_category"])
        l_rating    = self.loss_rating(outputs["rating"].squeeze(1), labels["label_rating"])
        l_priority  = self.loss_priority(outputs["priority"],  labels["label_priority"])

        total = (
            self.task_weights["sentiment"] * l_sentiment +
            self.task_weights["category"]  * l_category  +
            self.task_weights["rating"]    * l_rating    +
            self.task_weights["priority"]  * l_priority
        )

        return total, {
            "sentiment": l_sentiment.item(),
            "category":  l_category.item(),
            "rating":    l_rating.item(),
            "priority":  l_priority.item(),
        }


def train_epoch(model, loader: DataLoader, optimizer, criterion: MultiTaskLoss, device, scheduler=None):
    model.train()

    running = {"sentiment": 0.0, "category": 0.0, "rating": 0.0, "priority": 0.0}
    n_batches = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in batch.items()
                  if k.startswith("label_")}

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss, loss_parts = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        for k in running:
            running[k] += loss_parts[k]
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


def eval_epoch(model, loader: DataLoader, device):
    model.eval()

    all_sentiment_preds, all_sentiment_labels = [], []
    all_rating_preds,    all_rating_labels    = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch.items()
                      if k.startswith("label_")}

            outputs = model(input_ids, attention_mask)

            sentiment_preds = outputs["sentiment"].argmax(dim=1)
            all_sentiment_preds.extend(sentiment_preds.cpu().tolist())
            all_sentiment_labels.extend(labels["label_sentiment"].cpu().tolist())

            rating_preds = outputs["rating"].squeeze(1)
            all_rating_preds.extend(rating_preds.cpu().tolist())
            all_rating_labels.extend(labels["label_rating"].cpu().tolist())

    f1  = f1_score(all_sentiment_labels, all_sentiment_preds, average="macro")
    mae = mean_absolute_error(all_rating_labels, all_rating_preds)

    return {"f1_macro": f1, "mae": mae}
