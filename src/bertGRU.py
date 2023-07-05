import torch
import numpy as np
from transformers import BertModel
from torch import nn
import torch.nn.functional as F
from src.sent_att_model import SentAttNet

class BertClassifier(nn.Module):

    def __init__(self, bert_model, hidden_size, num_classes, batch_size, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout)
        self.sent_att_net = SentAttNet(hidden_size,self.bert_model.config.hidden_size,num_classes)
        self.sent_hidden_state = torch.zeros(2, batch_size, hidden_size)

      
    def forward(self, input):
        batch_embedding = []
        for document_ids in input:
            document_pooled_output = self.bert_model(input_ids=document_ids)
            document_pooled_output = torch.mean(document_pooled_output.last_hidden_state, dim=1)
            # document_pooled_output = document_pooled_output['pooler_output'].squeeze()
            batch_embedding.append(document_pooled_output)
        batch_embedding = torch.stack(batch_embedding)
        batch_embedding = batch_embedding.permute(1,0,2)
        output, self.sent_hidden_state = self.sent_att_net(batch_embedding, self.sent_hidden_state)
        return output

