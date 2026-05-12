import re
import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm

tqdm.pandas()

# ---------------------------------------------------------------------------
# Keyword dictionary — 8 categorias
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, list[str]] = {
    "limpeza": [
        "clean", "dirty", "spotless", "filthy", "hygiene",
        "dust", "mold", "stain", "tidy", "messy", "cleanliness", "smelly", "odor",
        "bathroom", "shower", "toilet", "immaculate", "grimy",
        "musty", "sanitary", "towel", "linen", "sheet",
    ],
    "atendimento": [
        "staff", "friendly", "rude", "helpful", "receptionist",
        "concierge", "check-in", "check-out", "front desk", "service",
        "employee", "host", "attitude", "courteous", "unprofessional",
        "management", "manager", "supervisor",
    ],
    "localização": [
        "location", "central", "downtown", "nearby", "walk",
        "metro", "airport", "area", "neighborhood", "distance",
        "situated", "close to", "far from", "accessible", "convenient",
    ],
    "alimentação": [
        "breakfast", "restaurant", "food", "meal", "dinner",
        "lunch", "buffet", "menu", "cuisine", "bar",
        "coffee", "drink", "snack", "kitchen", "dining",
    ],
    "preço": [
        "price", "expensive", "cheap", "worth",
        "cost", "money", "affordable", "overpriced", "fee",
        "rate", "budget", "deal", "pricey", "reasonable",
        "charge", "charged", "bill", "billed",
    ],
    "conforto": [
        "comfortable", "bed", "pillow", "mattress", "noise",
        "quiet", "cozy", "spacious", "small room", "cramped",
        "temperature", "air conditioning", "heating", "sleep", "relaxing",
    ],
    "wifi": [
        "wifi", "wi-fi", "internet", "connection",
        "network", "online", "bandwidth", "slow internet", "signal",
    ],
    "instalações": [
        "pool", "gym", "spa", "facilities", "elevator",
        "parking", "lobby", "amenities", "fitness", "sauna",
        "jacuzzi", "rooftop", "terrace", "laundry", "shuttle",
    ],
}

# ---------------------------------------------------------------------------
# Funções individuais
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def derive_sentiment(overall: float) -> str:
    if overall <= 2:
        return "negativo"
    if overall == 3:
        return "neutro"
    return "positivo"


def derive_categories(text: str) -> dict[str, int]:
    text_lower = text.lower()
    return {
        cat: int(any(kw in text_lower for kw in kws))
        for cat, kws in KEYWORDS.items()
    }


def derive_priority(sentiment: str, overall: float) -> str:
    if sentiment == "negativo" and overall <= 2:
        return "alta"
    return "normal"


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Limpar texto
    df["review_text"] = df["review_text"].map(clean_text)

    # 2. Filtrar idioma — manter apenas inglês
    print("Detectando idioma...")
    df["lang"] = df["review_text"].progress_map(detect_language)
    n_before = len(df)
    df = df[df["lang"] == "en"].drop(columns="lang").reset_index(drop=True)
    print(f"Filtro de idioma: {n_before - len(df):,} reviews removidos, {len(df):,} mantidos.")

    # 3. Labels de sentimento
    df["sentiment"] = df["overall"].map(derive_sentiment)

    # 4. Labels de categoria (multi-label)
    cats = df["review_text"].map(derive_categories)
    df = pd.concat([df, pd.DataFrame(cats.tolist(), index=df.index)], axis=1)

    # 5. Label de prioridade
    df["priority"] = df.apply(
        lambda row: derive_priority(row["sentiment"], row["overall"]), axis=1
    )

    return df
