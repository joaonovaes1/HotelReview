"""
2VA — Simulação C_alt: Modelo Único (30 seeds)
Roda as 30 simulações e salva results/single_model_results.csv
"""
import sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import HotelReviewClassifier
from src.models.trainer    import MultiTaskLoss, train_epoch, eval_epoch
from src.data.dataset      import ReviewDataset

# ── config ────────────────────────────────────────────────────────────────────
SEEDS        = list(range(30))
N_EPOCHS     = 2
BATCH_SIZE   = 16
LR           = 2e-5
VAL_SPLIT    = 0.2
DATA_PATH    = Path(__file__).parent / "data" / "sample_2000.csv"
RESULTS_PATH = Path(__file__).parent / "results" / "single_model_results.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device : {DEVICE}  |  dataset : {DATA_PATH.name}  |  epochs : {N_EPOCHS}")

# ── dataset (carregado uma vez) ───────────────────────────────────────────────
full_dataset = ReviewDataset(str(DATA_PATH))
n_val   = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loaders(seed):
    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(full_dataset, [n_train, n_val], generator=gen)
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False))


def run_single_model(seed):
    set_seed(seed)
    train_loader, val_loader = make_loaders(seed)

    model     = HotelReviewClassifier().to(DEVICE)
    criterion = MultiTaskLoss().to(DEVICE)          # fix: criterion no device correto
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = N_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for _ in range(N_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler)

    metrics = eval_epoch(model, val_loader, DEVICE)
    return {"seed": seed, "f1_macro": round(metrics["f1_macro"], 4),
            "mae": round(metrics["mae"], 4)}


# ── loop principal ────────────────────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results = []

for i, seed in enumerate(SEEDS):
    print(f"[{i+1:02d}/30] seed={seed} ...", end=" ", flush=True)
    row = run_single_model(seed)
    results.append(row)
    print(f"F1={row['f1_macro']:.4f}  MAE={row['mae']:.4f}")

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)
print(f"\nSalvo em {RESULTS_PATH}")
print(df[["f1_macro", "mae"]].describe().round(4).to_string())
