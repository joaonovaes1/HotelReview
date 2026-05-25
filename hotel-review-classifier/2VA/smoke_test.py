import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent               # hotel-review-classifier/
sys.path.insert(0, str(project_root))                     # resolve src.models.classifier
sys.path.insert(1, str(project_root / "2VA" / "src"))     # resolve ensemble diretamente

from ensemble import HotelReviewEnsemble

# --- setup: instância sem chamar __init__ (evita carregar checkpoints do disco) ---
ensemble = object.__new__(HotelReviewEnsemble)
ensemble.device = torch.device("cpu")
ensemble.models = []

# --- fake outputs: simula o que _forward_one retornaria para cada modelo ---
batch_size = 1
fake_outputs = [
    {
        "sentiment": torch.randn(batch_size, 3),
        "category":  torch.randn(batch_size, 8),
        "rating":    torch.randn(batch_size, 1),
        "priority":  torch.randn(batch_size, 2),
    }
    for _ in range(3)   # 3 modelos no ensemble
]

# --- Passo A: testa _aggregate ---
aggregated = ensemble._aggregate(fake_outputs)

assert aggregated["sentiment"].shape == (batch_size, 3),  "sentiment shape errado"
assert aggregated["category"].shape  == (batch_size, 8),  "category shape errado"
assert aggregated["rating"].shape    == (batch_size, 1),  "rating shape errado"
assert aggregated["priority"].shape  == (batch_size, 2),  "priority shape errado"

# softmax deve somar 1 por linha
assert aggregated["sentiment"].sum(dim=1).allclose(torch.ones(batch_size)), "sentiment não soma 1"
assert aggregated["priority"].sum(dim=1).allclose(torch.ones(batch_size)),  "priority não soma 1"

# sigmoid deve estar entre 0 e 1
assert (aggregated["category"] >= 0).all() and (aggregated["category"] <= 1).all(), "category fora de [0,1]"

print("[OK] _aggregate: shapes e ranges corretos")

# --- Passo B: testa decodificação (lógica do predict sem carregar modelos) ---
sentiment_idx  = aggregated["sentiment"].argmax(dim=1).item()
priority_idx   = aggregated["priority"].argmax(dim=1).item()
category_probs = aggregated["category"].squeeze(0)
rating_norm    = aggregated["rating"].squeeze().item()

result = {
    "sentiment":  HotelReviewEnsemble.SENTIMENT_LABELS[sentiment_idx],
    "priority":   HotelReviewEnsemble.PRIORITY_LABELS[priority_idx],
    "rating":     round(rating_norm * 4 + 1, 1),
    "categories": [
        HotelReviewEnsemble.CATEGORY_NAMES[i]
        for i, prob in enumerate(category_probs)
        if prob.item() > 0.5
    ],
}

assert result["sentiment"] in {"negativo", "neutro", "positivo"}, "sentimento inválido"
assert result["priority"]  in {"normal", "alta"},                 "prioridade inválida"
assert 1.0 <= result["rating"] <= 5.0,                            "rating fora de [1, 5]"
assert isinstance(result["categories"], list),                    "categories não é lista"

print(f"[OK] decodificacao: {result}")
print("[OK] Smoke test concluido — ensemble.py esta correto")
