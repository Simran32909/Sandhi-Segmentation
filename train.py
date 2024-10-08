import torch
import json
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from utils.collate_fn import pad_collate
from utils.data_loader import SandhiDataset
from models.sandhi_segmenter import SandhiSegmenter
from utils.preprocess import Tokenizer

def load_vocab(filepath='data/vocab.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

vocab = load_vocab('data/vocab.json')
tokenizer = Tokenizer(vocab)


EMB_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 30
PAD_IDX = 0


train_dataset = SandhiDataset('data/wsmp_train_1.json', tokenizer)
val_dataset = SandhiDataset('data/wsmp_dev_1.json', tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SandhiSegmenter(
    vocab_size=len(tokenizer.vocab),  # Use 'vocab_size' instead of 'input_dim'
    emb_dim=EMB_DIM,
    hidden_dim=128,
    output_dim=2,
    pad_idx=0  # Optional: If you have a padding index
)
pad_idx=0
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        # Forward pass
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', leave=False):
            outputs = model(inputs)
        # No loss computation, backpropagation or optimizer step
        # loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        # loss.backward()
        # optimizer.step()

torch.save(model.state_dict(), "model_checkpoint.pth")
