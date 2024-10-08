import  json
import torch
from torch.utils.data import Dataset

class SandhiDataset(Dataset):
    def __init__(self,file_path, tokenizer):
        self.data = json.load(open(file_path, encoding='utf-8'))
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item=self.data[idx]
        input_seq = self.tokenizer.text_to_sequence(item['joint_sentence'])
        target_seq = self.tokenizer.text_to_sequence(item['segmented_sentence'])
        return torch.tensor(input_seq), torch.tensor(target_seq)
