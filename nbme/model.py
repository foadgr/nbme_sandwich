import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = DistilBertConfig.from_pretrained(cfg.model)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = DistilBertModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = DistilBertModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.dim, 1)  # DistilBert uses "dim" instead of "hidden_size"
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        # DistilBERT only requires input_ids and attention_mask (no token type IDs)
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return outputs[0]  # the last hidden state

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output
