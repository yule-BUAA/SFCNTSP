import torch
import torch.nn as nn

from model.TemporalDependencies import TemporalDependencies
from model.ElementRelationships import ElementRelationships
from model.ChannelCorrelations import ChannelCorrelations
from model.RepresentationsFusing import RepresentationsFusing


class SFCNTSP(nn.Module):
    def __init__(self, num_items: int, max_seq_length: int, embedding_channels: int, dropout: float, bias: bool = False,
                 alpha: float = 1.0, beta: float = 0.1):

        super(SFCNTSP, self).__init__()

        self.num_items = num_items
        self.embedding_channels = embedding_channels

        self.dropout = nn.Dropout(dropout)

        self.items_embedding = nn.Parameter(torch.randn([num_items, embedding_channels]), requires_grad=True)

        self.temporal_dependencies = TemporalDependencies(in_size=max_seq_length, out_size=max_seq_length, dropout=dropout, bias=bias)

        self.element_relationships = ElementRelationships(dropout=dropout, alpha=alpha, beta=beta)

        self.channel_correlations = ChannelCorrelations(in_size=embedding_channels, out_size=embedding_channels, dropout=dropout, bias=bias)

        self.representations_fusing = RepresentationsFusing(num_items=num_items, embedding_channels=embedding_channels, dropout=dropout)

    def forward(self, batch_seq_length: list, batch_items_id: list, batch_set_size: list, batch_input_data: torch.tensor):

        # input_tensor, torch.Tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        input_tensor = self.dropout(self.items_embedding[batch_input_data])

        # hidden_tensor, torch.Tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        hidden_tensor = self.temporal_dependencies(input_tensor=input_tensor, batch_seq_length=batch_seq_length, batch_set_size=batch_set_size)

        # hidden_tensor, torch.Tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        hidden_tensor = self.element_relationships(input_tensor=hidden_tensor, batch_set_size=batch_set_size)

        # hidden_tensor, torch.Tensor, shape -> (batch_size, max_seq_len, max_set_size, embedding_channels)
        hidden_tensor = self.channel_correlations(input_tensor=hidden_tensor)

        # batch_set_predict shape -> (batch_size, num_items)
        batch_set_predict = self.representations_fusing(input_tensor=hidden_tensor,
                                                        batch_seq_length=batch_seq_length,
                                                        batch_items_id=batch_items_id,
                                                        items_embedding=self.items_embedding)

        return batch_set_predict
