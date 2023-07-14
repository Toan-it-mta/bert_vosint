from transformers import BertModel, BertTokenizer
import torch
from src.BertClassifier import BertClassifier
import numpy as np
import warnings 
import re
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_PATH ="models/0.76_best_model.pt"
BERT_NAME ="NlpHUST/vibert4news-base-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME)
sentiment_model = BertClassifier(bert_model,num_classes=3)
sentiment_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device("cpu")))
sentiment_model.eval()

MAX_SENT_LENGTH = 100
MAX_WORD_LENGTH = 100
LABEL_MAPPING = {0: "tieu_cuc",
                    1: "trung_tinh",
                    2: "tich_cuc"
                }

def preprocessing_text(text):
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("\.{2,}", ".", text)
    text.strip()
    return text

def preprocess(text:str):
    text = preprocessing_text(text)
    paragraphs = text.split("\n")
    sentences_ids = []
    sentences_mask = []
    sentences = []
    for paragraph in paragraphs:
        # lặp từng câu
        for sentence in sent_tokenize(text=paragraph):
            sentences.append(sentence)

    sentences_token = tokenizer(
        sentences, 
        max_length=MAX_WORD_LENGTH, 
        truncation=True, return_tensors="pt", 
        padding='max_length')
    
    sentences_ids = sentences_token['input_ids']
    sentences_mask = sentences_token['attention_mask']
    num_sent = len(sentences_ids)
    if len(sentences_ids) >= MAX_SENT_LENGTH:
        sentences_ids = sentences_ids[:MAX_SENT_LENGTH]
        sentences_mask = sentences_mask[:MAX_SENT_LENGTH]
        num_sent = MAX_SENT_LENGTH
    else:
        
        sentences_ids_padding = torch.zeros((MAX_SENT_LENGTH - len(sentences_ids),MAX_WORD_LENGTH),dtype=torch.long)
        sentences_ids = torch.concat((sentences_ids,sentences_ids_padding),0)
        sentences_mask_padding = torch.zeros((MAX_SENT_LENGTH - len(sentences_mask),MAX_WORD_LENGTH),dtype=torch.long)
        sentences_mask = torch.concat((sentences_mask,sentences_mask_padding),0)
    
    sentences_ids = sentences_ids.view(1,MAX_SENT_LENGTH,MAX_WORD_LENGTH)
    sentences_mask = sentences_mask.view(1,MAX_SENT_LENGTH,MAX_WORD_LENGTH)
    assert sentences_ids.size() == sentences_mask.size()
    return sentences_ids, sentences_mask, [num_sent]

def predict(text:str):
    with torch.no_grad():           
        sentences_ids ,sentences_mask, num_sent = preprocess(text)
        logits = sentiment_model(sentences_ids,sentences_mask,num_sent)
        logits = logits.cpu().detach().numpy()[0]
    index_pred = np.argmax(logits, -1)
    label_pred = LABEL_MAPPING[index_pred]
    return ["NO",label_pred]

if __name__ == "__main__":
    file_input = "/media/nlp_team/nlpteam/toan/V_OSINT_Sentiment_Analysis/dataset/SentimentTest/VnNewsConfirm.json"
    file_output = "/media/nlp_team/nlpteam/toan/V_OSINT_Sentiment_Analysis/dataset/SentimentTest/VnNewsConfirm_output_0.76_unpadding.json"
    with open(file_input,'r',encoding='utf-8') as f:
        dataset = json.load(f)

    for record in tqdm(dataset):
        txt = record.get('title',"")+'\n'
        txt += record.get('description',"")+'\n'
        txt += record.get('text')
        record['predict'] = predict(txt)

    with open(file_output,'w',encoding='utf-8') as f:
        dataset = json.dump(dataset,f,ensure_ascii=False)
    # text = """Người đứng đầu Bộ Quốc phòng tuyên bố rằng một thoả thuận hợp tác về mua sắm quốc phòng sẽ được ký với Bộ trưởng Quốc phòng Hoa Kỳ Lloyd Austin trong cuộc họp của họ vào thứ Sáu."""
    # predict_out = predict(text)
    # print(predict_out)

