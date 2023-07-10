import torch
import numpy as np
from transformers import BertModel
from torch import nn
import torch.nn.functional as F
from src.sent_att_model import SentAttNet

# Mô hình Bert + MLP
class BertClassifier(nn.Module):

    def __init__(self, bert_model, num_classes, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout)
        # print(self.bert_model.config.hidden_size(),' ',num_classes)
        self.linear = nn.Linear(768,num_classes)
        self.relu = nn.ReLU()

    def forward(self, ids, masks):
        batch_embedding = []
        batch_ids = ids
        batch_mask = masks
        for i in range(0,batch_ids.size(0)):
            document_ids = batch_ids[i]
            document_mask = batch_mask[i]
            _, document_pooled_output = self.bert_model(document_ids,document_mask,return_dict=False)
            document_pooled_output = torch.mean(document_pooled_output, dim=0)

            # document_pooled_output = document_pooled_output['pooler_output'].squeeze()
            batch_embedding.append(document_pooled_output)
        batch_embedding = torch.stack(batch_embedding)
        # batch_embedding = batch_embedding.permute(1,0,2)
        dropout_output = self.dropout(batch_embedding)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

