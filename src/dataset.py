import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
import re


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, dataset, label_mapping, max_token_length = 512, max_sent_length = 70):
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
        texts_token = []
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

            sentences_token = self.tokenizer(
                sentences, 
                max_length=max_token_length, 
                truncation=True, return_tensors="pt", 
                padding='max_length')
            
            sentences_ids = sentences_token['input_ids']
            sentences_mask = sentences_token['attention_mask']
            if len(sentences_ids) >= max_sent_length:
                sentences_ids = sentences_ids[:max_sent_length]
                sentences_mask = sentences_mask[:max_sent_length]
            else:
                
                sentences_ids_padding = torch.zeros((max_sent_length - len(sentences_ids),max_token_length),dtype=torch.long)
                sentences_ids = torch.concat((sentences_ids,sentences_ids_padding),0)
                sentences_mask_padding = torch.zeros((max_sent_length - len(sentences_mask),max_token_length),dtype=torch.long)
                sentences_mask = torch.concat((sentences_mask,sentences_mask_padding),0)
            assert sentences_ids.size() == sentences_mask.size()
            sentences_token['input_ids'] = sentences_ids
            sentences_token['attention_mask'] = sentences_mask
            texts_token.append(sentences_token)
        return texts_token

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
        batch_ids = self.get_batch_texts(idx)['input_ids']
        batch_mask = self.get_batch_texts(idx)['attention_mask']
        batch_y = self.get_batch_labels(idx)

        return batch_ids, batch_mask, batch_y
