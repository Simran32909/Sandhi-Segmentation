class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def tokenize(self, sentence):
        return list(sentence)

    def text_to_sequence(self, text):
        return [self.char2idx.get(char, self.char2idx['<pad>']) for char in text]

    def sequence_to_text(self, sequence):
        return ''.join([self.idx2char.get(idx, '') for idx in sequence if idx in self.idx2char])