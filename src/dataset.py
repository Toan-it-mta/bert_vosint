import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
import re


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, dataset, label_mapping, max_token_length = 100, max_sent_length = 70):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.labels = [label_mapping[record['label'][1]] for record in dataset]
        self.texts = []
        for record in dataset:
            txt = record['title'].strip()+record.get('description',
                                                     "").strip()+record['text'].strip()
            self.texts.append(txt)
        self.texts_encode = self.process(max_token_length, max_sent_length)

    def process(self,max_token_length,max_sent_length):
        # word_set_un = set()
        texts_encode = []
        for i in tqdm(range(0, len(self.texts))):
            # Thêm tách câu bằng dấu \n nữa
            text = self.texts[i]
            text = self.preprocessing_text(text)
            paragraphs = text.split("\n")
            sentences_ids = []
            sentences_mask = []
            sentences = []
            for paragraph in paragraphs:
                # lặp từng câu
                for sentence in sent_tokenize(text=paragraph):
                    sentences.append(sentence)

            # print("Do dai cau: ",len(sentences))   
            sentences_ids = self.tokenizer(
                sentences, 
                max_length=max_token_length, 
                truncation=True, return_tensors="pt", 
                padding='max_length')['input_ids']
            # sentences_ids.append(sen_encoding)
            if len(sentences_ids) >= max_sent_length:
                sentences_ids = sentences_ids[:max_sent_length]
            else:
                sentences_ids_padding = torch.zeros((max_sent_length - len(sentences_ids),max_token_length),dtype=torch.long)
                sentences_ids = torch.concat((sentences_ids,sentences_ids_padding),0)
            texts_encode.append(sentences_ids)
        texts_encode = torch.stack(texts_encode)
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
