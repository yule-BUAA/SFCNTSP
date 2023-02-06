import torch
import torch.nn as nn
from collections import defaultdict


class RepresentationsFusing(nn.Module):
    """
    Prediction Layer
    """

    def __init__(self, num_items: int, embedding_channels: int, dropout: float):

        super(RepresentationsFusing, self).__init__()

        self.num_items = num_items
        self.embedding_channels = embedding_channels

        self.dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(embedding_channels, embedding_channels)
        self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input_tensor: torch.Tensor, batch_seq_length: list, batch_items_id: list, items_embedding: nn.Parameter):

        batch_set_predict = []

        for batch_index, user_input_tensor in enumerate(input_tensor):
            # user_input_tensor, shape (user_seq_len, max_set_size, embedding_channels)
            user_input_tensor = user_input_tensor[:batch_seq_length[batch_index]]
            historical_item_dict = defaultdict(list)
            # user_seq_tensor, shape (max_set_size, embedding_channels)
            for seq_index, user_seq_tensor in enumerate(user_input_tensor):
                for item_index, item_id in enumerate(batch_items_id[batch_index][seq_index]):
                    historical_item_dict[item_id].append(user_seq_tensor[item_index])

            # shape, (num_items, embedding_channels)
            batch_all_items = self.dropout(items_embedding)

            # shape, (user_actual_items, embedding_channels)
            user_sequence_embedding = self.dropout(
                torch.stack([torch.mean(torch.stack(historical_item_dict[item_id], dim=0), dim=0)
                             for item_id in historical_item_dict], dim=0))

            # shape, (num_items, user_actual_items)
            user_items_attention_scores = torch.softmax(self.leaky_relu_func(torch.matmul(batch_all_items, user_sequence_embedding.t())), dim=-1)

            # shape, (num_items, embedding_channels)
            user_items_embedding = torch.matmul(user_items_attention_scores, user_sequence_embedding)

            # shape, (num_items)
            batch_set_predict.append((self.projection(user_items_embedding) * self.dropout(items_embedding)).sum(dim=-1))

        # shape, (batch_size, num_items)
        batch_set_predict = torch.stack(batch_set_predict, dim=0)

        return batch_set_predict
