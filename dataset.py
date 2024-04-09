import torch
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

PAD_TOKEN = 0

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD"}
        self.n_words = 1  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class ScanData(Dataset):
    def __init__(self, path, input_language=None, output_language=None):
        self.pairs = self.read_pairs(path)

        if not input_language and not output_language:
            self.input_language = Lang("INPUT")
            self.output_language = Lang("OUTPUT")
            for ip, op in self.pairs:
                self.input_language.addSentence(ip)
                self.output_language.addSentence(op)
        else:
            self.input_language = input_language
            self.output_language = output_language

    def read_pairs(self, path):
        lines = open(path, encoding="utf-8").read().strip().split("\n")
        pairs = []
        for line in lines:
            inl, outl = line.split("OUT:")
            inl = inl.replace("IN:", "").strip()
            outl = outl.strip()
            pairs.append([inl, outl])
        return pairs

    def __len__(self):
        return len(self.pairs)

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        input_tensor = self.indexesFromSentence(self.input_language, x)
        output_tensor = self.indexesFromSentence(self.output_language, y)
        return torch.tensor(input_tensor), torch.tensor(output_tensor), x, y


class CollateFunctor:
    def __init__(self):
        self.pad_id = PAD_TOKEN

    def __call__(self, sentences: list):
        source_ids, target_ids, source_str, target_str = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(source_ids)
        target_ids, target_mask = self.collate_sentences(target_ids)
        return source_ids, source_mask, target_ids, target_mask, source_str, target_str

    def collate_sentences(self, sentences: list):
        lengths = [sentence.size(0) for sentence in sentences]
        max_length = max(lengths)

        subword_ids = torch.stack([
            F.pad(sentence, (0, max_length - length), value=self.pad_id)
            for length, sentence in zip(lengths, sentences)
        ])
        attention_mask = subword_ids == self.pad_id

        return subword_ids, attention_mask
    
    
if __name__ == "__main__":
    train_data = ScanData("./data/simple_split/tasks_train_simple.txt")
    val_data = ScanData("./data/simple_split/tasks_test_simple.txt", input_language=train_data.input_language, output_language=train_data.output_language)
    random_pair = random.choice(train_data.pairs)
    v_random_pair = random.choice(val_data.pairs)
    print(random_pair)
    print(v_random_pair)
    print(val_data[1])



