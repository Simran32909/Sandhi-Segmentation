import json
import torch
import argparse

from torch.nn.utils.rnn import pad_sequence

from models.sandhi_segmenter import SandhiSegmenter
from utils.preprocess import Tokenizer

def load_vocab(filepath='data/wsmp_train_1.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def preprocess_input(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)  # Make sure this function exists
    indices = [tokenizer.char2idx.get(token, tokenizer.unk_idx) for token in tokens]  # Use `get` to handle unknown tokens
    padded_indices = pad_sequence(indices, padding_value=tokenizer.pad_idx)  # Adjust based on your method
    return padded_indices.unsqueeze(0)  # Add batch dimension

def predict(sentence, model, tokenizer):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_input(sentence, tokenizer)
        outputs = model(input_tensor)
        predicted_indices = torch.argmax(outputs, dim=-1).squeeze().tolist()  # Get the predicted indices
        segmented_output = [tokenizer.idx2char[idx] for idx in predicted_indices if idx != tokenizer.pad_idx]  # Exclude pad tokens
    return ' '.join(segmented_output)  # Join into a final output string

def main():
    parser = argparse.ArgumentParser(description="Segment Sanskrit sandhied sentences.")
    parser.add_argument('--sentence', type=str, required=True, help="The sandhied Sanskrit sentence to segment.")
    parser.add_argument('--model-path', type=str, default='model_checkpoint.pth',
                        help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    vocab = load_vocab('data/vocab.json')
    tokenizer = Tokenizer(vocab)
    model = SandhiSegmenter(
        vocab_size=len(tokenizer.vocab),  # Use 'vocab_size' instead of 'input_dim'
        emb_dim=256,
        hidden_dim=128,
        output_dim=2,
        pad_idx=0  # Optional: If you have a padding index
    )
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()


    def segment_sandhi(sentence, model):
        input_seq = tokenizer.text_to_sequence(sentence)
        input_tensor = torch.tensor(input_seq).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        output_seq = output.argmax(-1).squeeze().tolist()

        segmented_sentence = tokenizer.sequence_to_text(output_seq)

        return segmented_sentence
    segmented_sentence = segment_sandhi(args.sentence, model)
    print("Original:", args.sentence)
    print("Segmented:", segmented_sentence)


if __name__ == "__main__":
    main()
