"""
2VA — Simulação C: Ensemble (30 seeds)
Roda as 30 simulações e salva results/ensemble_results.csv
"""
import sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, mean_absolute_error

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(Path(__file__).parent / "src"))

from src.models.classifier import HotelReviewClassifier
from src.models.trainer    import MultiTaskLoss, train_epoch
from src.data.dataset      import ReviewDataset
from ensemble              import HotelReviewEnsemble

# ── config ────────────────────────────────────────────────────────────────────
SEEDS        = list(range(30))
N_MODELS     = 3
N_EPOCHS     = 2
BATCH_SIZE   = 16
LR           = 2e-5
VAL_SPLIT    = 0.2
DATA_PATH    = Path(__file__).parent / "data" / "sample_2000.csv"
TMP_DIR      = Path(__file__).parent / "results" / "tmp"
RESULTS_PATH = Path(__file__).parent / "results" / "ensemble_results.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"device : {DEVICE}  |  dataset : {DATA_PATH.name}  |  modelos : {N_MODELS}  |  epochs : {N_EPOCHS}")

# ── dataset ───────────────────────────────────────────────────────────────────
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


def eval_ensemble(ensemble, val_loader):
    all_sent_pred, all_sent_label = [], []
    all_rate_pred, all_rate_label = [], []

    for batch in val_loader:
        ids   = batch["input_ids"].to(ensemble.device)
        mask  = batch["attention_mask"].to(ensemble.device)
        outs  = [ensemble._forward_one(m, ids, mask) for m in ensemble.models]
        agg   = ensemble._aggregate(outs)

        all_sent_pred.extend(agg["sentiment"].argmax(dim=1).cpu().tolist())
        all_sent_label.extend(batch["label_sentiment"].tolist())
        all_rate_pred.extend(agg["rating"].squeeze(1).cpu().tolist())
        all_rate_label.extend(batch["label_rating"].tolist())

    return {
        "f1_macro": f1_score(all_sent_label, all_sent_pred, average="macro"),
        "mae":      mean_absolute_error(all_rate_label, all_rate_pred),
    }


def run_ensemble(seed):
    _, val_loader = make_loaders(seed)
    ckpt_paths = []

    for i in range(N_MODELS):
        sub_seed = seed * N_MODELS + i
        set_seed(sub_seed)
        train_loader, _ = make_loaders(sub_seed)

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

        ckpt = TMP_DIR / f"seed{seed}_m{i}.pt"
        torch.save(model.state_dict(), ckpt)
        ckpt_paths.append(str(ckpt))

    ens     = HotelReviewEnsemble(ckpt_paths, device=DEVICE)
    metrics = eval_ensemble(ens, val_loader)

    for p in ckpt_paths:
        Path(p).unlink()

    return {"seed": seed, "f1_macro": round(metrics["f1_macro"], 4),
            "mae": round(metrics["mae"], 4)}


# ── loop principal ────────────────────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results = []

for i, seed in enumerate(SEEDS):
    print(f"[{i+1:02d}/30] seed={seed} ({N_MODELS} modelos) ...", end=" ", flush=True)
    row = run_ensemble(seed)
    results.append(row)
    print(f"F1={row['f1_macro']:.4f}  MAE={row['mae']:.4f}")

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)
print(f"\nSalvo em {RESULTS_PATH}")
print(df[["f1_macro", "mae"]].describe().round(4).to_string())
