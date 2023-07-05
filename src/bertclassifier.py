import torch
import numpy as np
from transformers import BertModel
from torch import nn


class BertClassifier(nn.Module):

    def __init__(self, bert_model, hidden_size, num_classes, batch_size, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.bert_model.config.hidden_size,
                          hidden_size, batch_first = True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.hidden_state = torch.zeros(1, batch_size, hidden_size)
        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.cuda()
        self.relu = nn.ReLU()

    def forward(self, input):
        batch_embedding = []
        for document_ids in input:
            document_pooled_output = self.bert_model(input_ids=document_ids)
            document_pooled_output = torch.mean(document_pooled_output.last_hidden_state, dim=1)
            # document_pooled_output = document_pooled_output['pooler_output'].squeeze()
            batch_embedding.append(document_pooled_output)
        batch_embedding = torch.stack(batch_embedding)
        dropout_output = self.dropout(batch_embedding)
        _, h_output = self.gru(dropout_output, self.hidden_state)
        linear_output = self.linear(h_output.squeeze(0))
        final_layer = self.relu(linear_output)

        return final_layer
