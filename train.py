from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from src.dataset import Dataset
from src.bertclassifier import BertClassifier
import json

train_path = "dataset\\test.json"
with open(train_path, encoding='utf-8') as f:
    train_dataset = json.load(f)

test_path = "dataset\\test.json"
with open(test_path, encoding='utf-8') as f:
    test_dataset = json.load(f)

# Tạo mô hình BERT
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Thiết lập tham số
hidden_size = 128
num_classes = 2
batch_size = 2
label_mapping = {'tieu_cuc': 0,
                 'trung_tinh': 1,
                 'tich_cuc': 2
                 }
# Tạo mô hình phân lớp document
model = BertClassifier(bert_model, hidden_size, num_classes, batch_size)

# Định nghĩa hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Huấn luyện mô hình


def train_model(model, train_data, num_epochs):
    model.train()
    train = Dataset(tokenizer, train_data, label_mapping)
    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for document_encode, labels in train_dataloader:
            optimizer.zero_grad()
            print('here')
            # Forward pass
            logits = model(document_encode)
            print('oke')
            # Tính toán loss và backpropagation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # In thông tin loss sau mỗi epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')


train_model(model, train_dataset, 2)
