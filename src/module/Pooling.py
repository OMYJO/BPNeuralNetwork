from torch import nn
from transformers import BertConfig
from transformers.modeling_utils import PreTrainedModel


class SequencePoolingV0(nn.Module):
    def forward(self, input):
        return input[0]


class MLMPoolingV0(PreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input):
        output = self.linear(input)
        return output


class WoLPoolingV0(PreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, input):
        output = self.linear(input)
        return output
