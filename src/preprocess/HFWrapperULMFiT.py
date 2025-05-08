import sys
import pathlib
from transformers import PreTrainedModel, BertConfig
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.preprocess.ULMFiTClassifier import ULMFiTClassifier

class HFWrapperULMFiT(PreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.ulmfit_model = ULMFiTClassifier(model_name=config._name_or_path)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.ulmfit_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}