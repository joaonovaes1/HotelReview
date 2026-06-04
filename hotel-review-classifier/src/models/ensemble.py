import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.models.classifier import HotelReviewClassifier


class EnsembleInference:

    SENTIMENT_LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}
    PRIORITY_LABELS  = {0: "normal",   1: "alta"}
    CATEGORY_NAMES   = [
        "limpeza", "atendimento", "localização",
        "alimentação", "preço", "conforto", "wifi", "instalações",
    ]

    def __init__(self, checkpoint_paths: list, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.models = []
        for path in checkpoint_paths:
            m = HotelReviewClassifier()
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.to(self.device)
            m.eval()
            self.models.append(m)

    def _aggregate(self, outputs_list: list) -> dict:
        sentiment = torch.stack([F.softmax(o["sentiment"], dim=1) for o in outputs_list]).mean(0)
        priority  = torch.stack([F.softmax(o["priority"],  dim=1) for o in outputs_list]).mean(0)
        category  = torch.stack([torch.sigmoid(o["category"]) for o in outputs_list]).mean(0)
        rating    = torch.stack([o["rating"] for o in outputs_list]).mean(0)
        return {"sentiment": sentiment, "priority": priority, "category": category, "rating": rating}

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs_list = [m(input_ids, attention_mask) for m in self.models]

        agg = self._aggregate(outputs_list)

        sentiment_idx  = agg["sentiment"].argmax(dim=1).item()
        priority_idx   = agg["priority"].argmax(dim=1).item()
        category_probs = agg["category"].squeeze(0)
        rating_norm    = agg["rating"].squeeze().item()

        return {
            "sentiment":  self.SENTIMENT_LABELS[sentiment_idx],
            "priority":   self.PRIORITY_LABELS[priority_idx],
            "rating":     round(rating_norm * 4 + 1, 1),
            "categories": [
                self.CATEGORY_NAMES[i]
                for i, prob in enumerate(category_probs)
                if prob.item() > 0.5
            ],
        }
