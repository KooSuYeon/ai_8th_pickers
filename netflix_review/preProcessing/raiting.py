import pandas as pd
# 전처리 함수
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator 


from torchtext.data.utils import get_tokenizer
# 텍스트 전처리와 자연어 처리를 위한 라이브러리
from textblob import TextBlob

# 감성 분석을 위한 함수
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity



def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  # 대문자를 소문자로
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.strip()  # 띄어쓰기 제외하고 빈 칸 제거
    return text

def yield_tokens(sentences, tokenizer):
    for text in sentences:
        yield tokenizer(text)


def getSplitReviewsAndRaitings(data):
    data["content"] = data["content"].apply(preprocess_text)
    X = data["content"].tolist()
    y = data["score"].tolist()

    train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(X, y, test_size=0.2, random_state=42)

    return train_reviews, test_reviews, train_ratings, test_ratings

