import torch
from transformers import AutoTokenizer
from src.models.classifier import HotelReviewClassifier


class ReviewInference:

    SENTIMENT_LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}
    PRIORITY_LABELS  = {0: "normal",   1: "alta"}
    CATEGORY_NAMES   = [
        "limpeza", "atendimento", "localização",
        "alimentação", "preço", "conforto", "wifi", "instalações",
    ]

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = HotelReviewClassifier()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

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
            outputs = self.model(input_ids, attention_mask)

        sentiment_idx = outputs["sentiment"].argmax(dim=1).item()
        priority_idx  = outputs["priority"].argmax(dim=1).item()
        category_probs = torch.sigmoid(outputs["category"]).squeeze(0)
        rating_norm    = outputs["rating"].squeeze().item()

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
