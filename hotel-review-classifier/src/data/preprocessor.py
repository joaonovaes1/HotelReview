import re
import pandas as pd
from langdetect import detect, LangDetectException


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\sÀ-ɏḀ-ỿ.,!?;:()\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in ("pt", "en") else "en"
    except LangDetectException:
        return "en"


def combine_reviews(row: pd.Series) -> str:
    pos = clean_text(str(row.get("Positive_Review", "")))
    neg = clean_text(str(row.get("Negative_Review", "")))
    parts = [p for p in [pos, neg] if p and p.lower() not in ("no positive", "no negative", "nothing", "none")]
    return " ".join(parts)


def derive_sentiment(score: float) -> int:
    if score <= 5:
        return 0  # negativo
    elif score <= 7:
        return 1  # neutro
    else:
        return 2  # positivo


def derive_priority(sentiment_label: int, score: float) -> int:
    return 1 if (sentiment_label == 0 and score < 6) else 0


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df.apply(combine_reviews, axis=1)
    df["language"] = df["text"].apply(detect_language)
    df["sentiment_label"] = df["Reviewer_Score"].apply(derive_sentiment)
    df["rating_normalized"] = df["Reviewer_Score"] / 10.0
    df["priority_label"] = df.apply(
        lambda r: derive_priority(r["sentiment_label"], r["Reviewer_Score"]), axis=1
    )
    return df


CATEGORY_KEYWORDS = {
    "limpeza": ["clean", "dirty", "hygiene", "limpeza", "sujo", "limpo", "higiene", "smell", "cheiro"],
    "atendimento": ["staff", "service", "reception", "helpful", "rude", "friendly", "atendimento", "recepção", "funcionários", "amável", "grosseiro"],
    "localização": ["location", "central", "close", "far", "metro", "beach", "localização", "localizado", "perto", "longe", "centro", "praia"],
    "alimentação": ["breakfast", "food", "restaurant", "meal", "buffet", "café", "café da manhã", "comida", "restaurante", "refeição"],
    "preço": ["price", "value", "expensive", "cheap", "worth", "preço", "valor", "caro", "barato", "custo"],
    "conforto": ["comfortable", "bed", "pillow", "mattress", "noise", "quiet", "confortável", "cama", "travesseiro", "colchão", "barulho", "silencioso"],
    "wifi": ["wifi", "internet", "connection", "slow", "fast", "conexão", "lento", "rápido"],
    "instalações": ["pool", "gym", "spa", "parking", "elevator", "facility", "piscina", "academia", "estacionamento", "elevador", "instalações"],
}


def classify_categories(text: str) -> list[str]:
    text_lower = text.lower()
    return [cat for cat, kws in CATEGORY_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def add_category_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categories = list(CATEGORY_KEYWORDS.keys())
    cat_lists = df["text"].apply(classify_categories)
    for cat in categories:
        df[f"cat_{cat}"] = cat_lists.apply(lambda cats: int(cat in cats))
    return df
