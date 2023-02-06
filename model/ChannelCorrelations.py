import torch
import torch.nn as nn


class ChannelCorrelations(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float, bias: bool = False):

        super(ChannelCorrelations, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)

        self.sfcn = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, input_tensor: torch.Tensor):

        hidden_tensor = input_tensor

        # output_tensor, shape -> (batch_size, embedding_channels, max_seq_len)
        output_tensor = self.sfcn(hidden_tensor)

        output_tensor += input_tensor

        output_tensor = self.dropout(output_tensor)

        return output_tensor
