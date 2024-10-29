import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        batch_size, seq_len, _ = lstm_output.size()
        hidden = lstm_output[:, -1, :]

        score = self.Va(torch.tanh(self.Wa(lstm_output) + self.Ua(hidden).unsqueeze(1)))
        attention_weights = torch.softmax(score, dim=1)

        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector