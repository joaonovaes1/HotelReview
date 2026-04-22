import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm


CATEGORIES = ["limpeza", "atendimento", "localização", "alimentação", "preço", "conforto", "wifi", "instalações"]
SENTIMENT_LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}
PRIORITY_LABELS = {0: "normal", 1: "alta"}
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"


class HotelReviewClassifier(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base", num_categories: int = 8):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.head_sentiment = nn.Linear(hidden, 3)
        self.head_category = nn.Linear(hidden, num_categories)
        self.head_rating = nn.Linear(hidden, 1)
        self.head_priority = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token

        return {
            "sentiment": self.head_sentiment(pooled),
            "category": self.head_category(pooled),
            "rating": self.head_rating(pooled).squeeze(-1),
            "priority": self.head_priority(pooled),
        }


class ReviewInference:
    def __init__(self, model_name: str = "xlm-roberta-base", checkpoint: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HotelReviewClassifier(model_name)

        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    def _tokenize(self, texts: list[str], max_length: int = 256):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        enc = self._tokenize(texts)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)

        sentiments = out["sentiment"].argmax(dim=-1).cpu().tolist()
        priorities = out["priority"].argmax(dim=-1).cpu().tolist()
        ratings = (out["rating"].cpu() * 10).clamp(0, 10).tolist()
        cat_probs = torch.sigmoid(out["category"]).cpu().tolist()

        results = []
        for i, text in enumerate(texts):
            cats = [CATEGORIES[j] for j, p in enumerate(cat_probs[i]) if p > 0.5]
            results.append({
                "text": text,
                "sentiment": SENTIMENT_LABELS[sentiments[i]],
                "sentiment_id": sentiments[i],
                "categories": cats if cats else ["geral"],
                "rating_predicted": round(ratings[i], 1),
                "priority": PRIORITY_LABELS[priorities[i]],
                "priority_id": priorities[i],
            })
        return results

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "text", batch_size: int = 32) -> pd.DataFrame:
        texts = df[text_col].fillna("").tolist()
        all_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Classificando reviews"):
            batch = texts[i: i + batch_size]
            all_results.extend(self.predict_batch(batch))

        result_df = pd.DataFrame(all_results)
        return pd.concat([df.reset_index(drop=True), result_df.drop(columns=["text"])], axis=1)


def load_inference_engine(checkpoint: str = None) -> ReviewInference:
    ckpt = checkpoint or (str(MODEL_DIR / "best_model.pt") if (MODEL_DIR / "best_model.pt").exists() else None)
    return ReviewInference(checkpoint=ckpt)
