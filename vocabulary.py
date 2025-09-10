import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import threshold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import spacy
from torch.nn.utils.rnn import pad_sequence
import os

class Vocabulary:
    def __init__(self, frequency_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.frequency_threshold = frequency_threshold
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
        for word, count in frequencies.items():
            if count >= self.frequency_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word

                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

import json

def main():
    # path to your text file
    file_path = "sentences.txt"

    # read file
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences if s.strip()]  # remove empty lines


    vocab = Vocabulary(frequency_threshold=2)  
    vocab.build_vocabulary(sentences)

    
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=4)

    print("Vocabulary saved to vocab.json")

if __name__ == "__main__":
    main()