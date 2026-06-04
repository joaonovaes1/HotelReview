import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, mean_absolute_error, classification_report


class MultiTaskLoss(nn.Module):

    def __init__(self, task_weights: dict = None):
        super().__init__()

        # pesos por tarefa que é o quanto cada loss contribui para o total
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

    f1     = f1_score(all_sentiment_labels, all_sentiment_preds, average="macro")
    mae    = mean_absolute_error(all_rating_labels, all_rating_preds)
    report = classification_report(
        all_sentiment_labels, all_sentiment_preds,
        target_names=["negativo", "neutro", "positivo"],
    )

    return {"f1_macro": f1, "mae": mae, "report": report}


def train_single_model(model, train_loader, val_loader, device, n_epochs=3, lr=2e-5, seed=42):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model.to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           steps_per_epoch=len(train_loader),
                           epochs=n_epochs)

    best_f1, best_state = 0.0, None
    for epoch in range(n_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        metrics = eval_epoch(model, val_loader, device)
        if metrics["f1_macro"] > best_f1:
            best_f1   = metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_f1


def train_ensemble(model_factory, train_loader, val_loader, device, save_dir,
                   n_models=3, base_seed=0, n_epochs=3, lr=2e-5):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for i in range(n_models):
        print(f"[ensemble] treinando modelo {i+1}/{n_models} (seed={base_seed + i})...")
        model = model_factory()
        state, f1 = train_single_model(
            model, train_loader, val_loader, device, n_epochs, lr, seed=base_seed + i
        )
        path = os.path.join(save_dir, f"model_{i}.pt")
        torch.save(state, path)
        results.append({"model": i, "seed": base_seed + i, "f1_macro": round(f1, 4), "path": path})
        print(f"           F1={f1:.4f} → salvo em {path}")
    return results
