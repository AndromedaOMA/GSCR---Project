import sys
import pathlib
from transformers import BertModel, PreTrainedModel, BertConfig
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.preprocess.ULMFiTClassifier import ULMFiTClassifier

# Set on config before HFWrapperULMFiT(config) when loading a local checkpoint whose
# weights fully replace the encoder (avoids redundant Hub loads that hit meta-device paths).
ENCODER_FROM_STRUCTURE_ONLY_ATTR = "_encoder_from_structure_only"


class HFWrapperULMFiT(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__(config)
        if getattr(config, ENCODER_FROM_STRUCTURE_ONLY_ATTR, False):
            self.ulmfit_model = ULMFiTClassifier(encoder=BertModel(config))
        else:
            self.ulmfit_model = ULMFiTClassifier(model_name=config._name_or_path)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Ignore token_type_ids and other BertTokenizer batch keys not used here.
        logits = self.ulmfit_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}