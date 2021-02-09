from torch import nn


class SequencePoolingV0(nn.Module):
    def forward(self, input):
        return input[0]
