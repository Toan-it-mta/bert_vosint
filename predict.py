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

MODEL_PATH ="models/0.7395_best_model.pt"
BERT_NAME ="NlpHUST/vibert4news-base-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME)
sentiment_model = BertClassifier(bert_model,num_classes=3)
sentiment_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device("cpu")))
sentiment_model.eval()

MAX_SENT_LENGTH = 50
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
    # file_input = "/media/nlp_team/nlpteam/toan/v_osint_topic_sentiment/SentimentTest/VnNewsConfirm.json"
    # file_output = "/media/nlp_team/nlpteam/toan/v_osint_topic_sentiment/SentimentTest/VnNewsConfirm_output_padding.json"
    # with open(file_input,'r',encoding='utf-8') as f:
    #     dataset = json.load(f)

    # for record in tqdm(dataset):
    #     txt = record.get('title',"")+'\n'
    #     txt += record.get('description',"")+'\n'
    #     txt += record.get('text')
    #     record['predict'] = predict(txt)

    # with open(file_output,'w',encoding='utf-8') as f:
    #     dataset = json.dump(dataset,f,ensure_ascii=False)
    text = """Nhà máy chở điện' di động trên biển
NHẬT BẢNTàu Battery Tanker X trang bị 96 module pin lithium sắt phosphate, có thể chở 241 MWh điện sạch với tầm hoạt động 300 km.

Mô phỏng tàu chở điện Battery Tanker X. Ảnh: PowerX
Mô phỏng tàu chở điện Battery Tanker X. Ảnh: PowerX

Công ty Nhật Bản PowerX đang phát triển "nhà máy điện di động" dưới dạng một tàu pin dài 140 m, vận chuyển 241 MWh năng lượng tái tạo qua khoảng cách ngắn trên biển, New Atlas hôm 30/5 đưa tin. Năng lượng tái tạo thường được sản xuất ở nơi cách khu vực cần điện khá xa. Do đó, tàu thủy trang bị hàng loạt bộ pin sẽ giúp chở điện đến đích một cách dễ dàng.

PowerX cho biết, Nhật Bản có biển sâu bao quanh và dễ xảy ra động đất, gây khó khăn khi sử dụng cáp truyền tải điện. Bên cạnh đó, giải pháp vận chuyển điện bằng tàu thủy giúp khắc phục các vấn đề như thời gian chết (khi cáp biển gặp trục trặc và cần sửa chữa) kéo dài, chi phí cho việc kết nối điện áp siêu cao và trạm biến áp lớn.

Thiết kế của nguyên mẫu tàu chở điện mang tên Battery Tanker X, được PowerX hé lộ tại Triển lãm Hàng hải Quốc tế Bariship diễn ra ở thành phố Imabari, tỉnh Ehime, Nhật Bản, hôm 29/5. Nguyên mẫu này trang bị 96 module pin lithium sắt phosphate với kích thước tương đương container. Battery Tanker X chạy bằng điện với tầm hoạt động tối đa dự kiến là 300 km. Tàu chở điện cũng sẽ trang bị các hệ thống kiểm soát khí thải và dập lửa.

PowerX cũng đang phát triển một phiên bản khác mang tên Power Ark với kích thước lớn hơn nhiều so với Battery Tanker X. Power Ark dự kiến trang bị lượng lithium gấp 8 lần, nghĩa là chở được tới 2 GWh điện - đủ cho khoảng 70.000 gia đình trung bình ở Mỹ dùng cả ngày.

PowerX đang thành lập một công ty con mới mang tên Ocean Power Grid nhằm thương mại hóa công nghệ mới. Với việc hoàn tất thiết kế chi tiết của nguyên mẫu, công ty đặt mục tiêu chế tạo Battery Tanker X vào năm 2025. Việc thử nghiệm thực địa trong nước và quốc tế dự kiến bắt đầu vào năm 2026."""
    predict_out = predict(text)
    print(predict_out)

