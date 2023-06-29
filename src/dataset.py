import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
import re


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, dataset, label_mapping):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.labels = [label_mapping[record['label'][1]] for record in dataset]
        self.texts = []
        for record in dataset:
            txt = record['title'].strip()+record.get('description',
                                                     "").strip()+record['text'].strip()
            self.texts.append(txt)
        self.texts_encode = self.process()

    def process(self):
        # word_set_un = set()
        texts_encode = []
        for i in tqdm(range(0, len(self.texts))):
            # Thêm tách câu bằng dấu \n nữa
            text = self.texts[i]
            text = self.preprocessing_text(text)
            paragraphs = text.split("\n")
            sentences_ids = []
            sentences_mask = []
            for paragraph in paragraphs:
                # lặp từng câu
                for sentence in sent_tokenize(text=paragraph):
                    sen_encoding = self.tokenizer(
                        sentence, max_length=100, truncation=True, return_tensors="pt", padding='max_length')
                    sentences_ids.append(sen_encoding['input_ids'].squeeze())
                    sentences_mask.append(
                        sen_encoding['attention_mask'].squeeze())
            sentences_ids = torch.stack(sentences_ids)
            sentences_mask = torch.stack(sentences_mask)
            texts_encode.append([sentences_ids, sentences_mask])
        return texts_encode

    def preprocessing_text(self, text):
        text = re.sub("\r", "\n", text)
        text = re.sub("\n{2,}", "\n", text)
        text = re.sub("…", ".", text)
        text = re.sub("\.{2,}", ".", text)
        text.strip()
        return text

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts_encode[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
