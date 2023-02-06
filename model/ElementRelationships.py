import torch
import torch.nn as nn


class ElementRelationships(nn.Module):
    def __init__(self, dropout: float, alpha: float = 1.0, beta: float = 0.1):

        super(ElementRelationships, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_tensor: torch.Tensor, batch_set_size: list):

        # batch_scores, shape -> (batch_size, seq_len, max_set_size, max_set_size)
        batch_scores = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
                                   input_tensor.shape[2]).to(input_tensor.device)

        # assign importance of each element in the set
        for batch_index, user_set_size in enumerate(batch_set_size):
            for set_index, set_size in enumerate(user_set_size):
                batch_scores[batch_index, set_index, :set_size, :set_size] = self.beta
                batch_scores[batch_index, set_index, :set_size, :set_size] += torch.eye(set_size).to(input_tensor.device) * self.alpha

        # output_tensor, shape -> (batch_size, seq_len, max_set_size, embedding_channels)
        output_tensor = torch.einsum('btnn,btnf->btnf', batch_scores, input_tensor)

        # output_tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        output_tensor = self.dropout(output_tensor)

        return output_tensor
