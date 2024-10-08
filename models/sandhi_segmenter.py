import torch
import torch.nn as nn

class SandhiSegmenter(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, output_dim=2, pad_idx=0):
        super(SandhiSegmenter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = self.fc(x)
        return x
