import torch
import torch.nn as nn
import torch.optim as optim

import os
import importlib.util

# 모듈을 동적으로 불러오는 함수
def load_module_from_path(path):
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

attention_path = "model_class/Attention.py"
Attention = load_module_from_path(attention_path)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = Attention.Attention(hidden_dim) 
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)  
        context_vector = self.attention(output)
        return self.fc(context_vector)
