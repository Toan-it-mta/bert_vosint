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
                          hidden_size, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, num_classes)
        self.hidden_state = torch.zeros(2, batch_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, sentences_encode):
        sentence_embeddings = []
        sentences_ids = sentences_encode[0]
        sentences_mask = sentences_encode[1]
        print('oke')
        for sentence_id, sentence_mask in zip(sentences_ids, sentences_mask):
            _, sentence_pooled_output = self.bert_model(
                input_ids=sentence_id, attention_mask=sentence_mask, return_dict=False)
            sentence_embeddings.append(sentence_pooled_output)
            print('oke 1')
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)

        dropout_output = self.dropout(sentence_embeddings)

        _, h_output = self.gru(dropout_output, self.hidden_state)
        linear_output = self.linear(h_output)
        final_layer = self.relu(linear_output)

        return final_layer
