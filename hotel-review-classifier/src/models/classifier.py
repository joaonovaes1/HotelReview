import torch.nn as nn
from transformers import AutoModel


class HotelReviewClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("xlm-roberta-base")
        self.head_sentiment = nn.Linear(768, 3)
        self.head_category = nn.Linear(768, 8)
        self.head_rating = nn.Linear(768, 1)
        self.head_priority = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :]

        return {
            "sentiment": self.head_sentiment(cls),
            "category":  self.head_category(cls),
            "rating":    self.head_rating(cls),
            "priority":  self.head_priority(cls),
        }
