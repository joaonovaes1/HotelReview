import torch
from src.models.classifier import HotelReviewClassifier


class HotelReviewEnsemble:

    SENTIMENT_LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}
    PRIORITY_LABELS  = {0: "normal",   1: "alta"}
    CATEGORY_NAMES   = [
        "limpeza", "atendimento", "localização",
        "alimentação", "preço", "conforto", "wifi", "instalações",
    ]

    def __init__(self, checkpoint_paths: list[str], device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models = []
        for path in checkpoint_paths:
            model = HotelReviewClassifier()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def _forward_one(self, model: HotelReviewClassifier, input_ids, attention_mask) -> dict:
        with torch.no_grad():
            return model(input_ids, attention_mask)

    def _aggregate(self, outputs_list: list[dict]) -> dict:
        # cada outputs_list[i] é um dict com tensores shape (batch, N_classes)
        # stack cria dimensão extra → (n_models, batch, N_classes), média colapsa ela
        sentiment_probs = torch.stack(
            [torch.softmax(o["sentiment"], dim=1) for o in outputs_list]
        ).mean(dim=0)

        category_probs = torch.stack(
            [torch.sigmoid(o["category"]) for o in outputs_list]
        ).mean(dim=0)

        rating_mean = torch.stack(
            [o["rating"] for o in outputs_list]
        ).mean(dim=0)

        priority_probs = torch.stack(
            [torch.softmax(o["priority"], dim=1) for o in outputs_list]
        ).mean(dim=0)

        return {
            "sentiment": sentiment_probs,
            "category":  category_probs,
            "rating":    rating_mean,
            "priority":  priority_probs,
        }

    def predict(self, input_ids, attention_mask) -> dict:
        outputs_list = [
            self._forward_one(model, input_ids, attention_mask)
            for model in self.models
        ]
        aggregated = self._aggregate(outputs_list)

        sentiment_idx  = aggregated["sentiment"].argmax(dim=1).item()
        priority_idx   = aggregated["priority"].argmax(dim=1).item()
        category_probs = aggregated["category"].squeeze(0)
        rating_norm    = aggregated["rating"].squeeze().item()

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
