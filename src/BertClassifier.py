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
        self.linear = nn.Linear(768,num_classes)
        self.relu = nn.ReLU()

    def forward(self, ids, masks, num_sents):
        batch_embedding = []
        for i in range(0,ids.size(0)):
            num_sent = num_sents[i]
            document_ids = ids[i][:num_sent]
            document_mask = masks[i][:num_sent]
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                document_ids = document_ids.to(device)
                document_mask =document_mask.to(device)
            
            _, document_pooled_output = self.bert_model(document_ids,document_mask,return_dict=False)
            # Chỉ lấy trung bình của số câu
            document_pooled_output = torch.mean(document_pooled_output, dim=0)
            batch_embedding.append(document_pooled_output)

        batch_embedding = torch.stack(batch_embedding)
        dropout_output = self.dropout(batch_embedding)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

