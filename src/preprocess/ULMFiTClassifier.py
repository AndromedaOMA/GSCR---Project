import torch
import torch.nn as nn
from transformers import AutoModel

class ULMFiTClassifier(nn.Module):
    def __init__(self, model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1", dropout: float = 0.15, num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Classifier Head (ULMFiT-style)
        self.pooler = lambda x: torch.cat(
            (
                x[:, 0],                           # CLS token
                torch.mean(x, dim=1),              # Mean pooling
                torch.max(x, dim=1).values         # Max pooling
            ),
            dim=1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        # For more in depth training
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size * 3, hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size, num_labels)
        # )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, T, H]
        pooled = self.pooler(sequence_output)        # [B, 3H]
        logits = self.fc(pooled)                     # [B, num_labels]
        return logits