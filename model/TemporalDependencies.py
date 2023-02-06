import torch
import torch.nn as nn


class TemporalDependencies(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float, bias: bool = False):

        super(TemporalDependencies, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)

        self.sfcn = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, input_tensor: torch.Tensor, batch_seq_length: list, batch_set_size: list):

        # use mask to filter the padded time slots
        batch_sequence_mask = torch.zeros(input_tensor.shape[:2]).to(input_tensor.device)

        for batch_index, user_seq_length in enumerate(batch_seq_length):
            batch_sequence_mask[batch_index, :user_seq_length] = 1

        hidden_tensor = torch.einsum('bt,btnf->btnf', batch_sequence_mask, input_tensor)

        batch_permute_invariant_mask = torch.zeros(input_tensor.shape[:3]).to(input_tensor.device)

        # use mask to filter the padded elements
        for batch_index, user_set_size in enumerate(batch_set_size):
            for set_index, set_size in enumerate(user_set_size):
                batch_permute_invariant_mask[batch_index, set_index, :set_size] = 1.0 / set_size

        hidden_tensor = torch.einsum('btn,btnf->btf', batch_permute_invariant_mask, hidden_tensor)

        # hidden_tensor, shape -> (batch_size, embedding_channels, max_seq_len)
        hidden_tensor = hidden_tensor.permute(0, 2, 1)

        # output_tensor, shape -> (batch_size, embedding_channels, max_seq_len)
        output_tensor = self.sfcn(hidden_tensor)

        # output_tensor, shape -> (batch_size, max_seq_len, embedding_channels)
        output_tensor = output_tensor.permute(0, 2, 1)

        # output_tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        output_tensor = output_tensor.unsqueeze(dim=2) + input_tensor

        output_tensor = self.dropout(output_tensor)

        return output_tensor
