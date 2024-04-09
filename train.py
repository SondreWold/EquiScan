import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import ScanData, CollateFunctor
from tqdm import tqdm
import math
import logging
import numpy as np
import random
from model import Transformer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyTransformer(nn.Module):
    def __init__(self, vocab_size_input=15, vocab_size_output=7,
     num_encoder_layers=6, num_decoder_layers=6, hidden_size=256, num_heads=4, dropout=0.1):
        super(MyTransformer, self).__init__()
        self.emb_in = nn.Embedding(vocab_size_input, hidden_size)
        self.emb_out = nn.Embedding(vocab_size_output, hidden_size)
        self.transformer_model = Transformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.projection = nn.Linear(hidden_size, vocab_size_output)
        self.init_weights()

    
    def init_weights(self) -> None:
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.emb_in(src)
        tgt = self.emb_out(tgt)
        out = self.transformer_model(src, src_mask, tgt, tgt_mask)
        out = self.projection(out)
        out = torch.softmax(out, dim=-1)
        return out

def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(0)
    logging.info("-----EquiScan------")
    batch_size = 32
    EPOCHS = 10
    HEADS = 4

    train_data = ScanData("./data/simple_split/tasks_train_simple.txt")
    val_data = ScanData("./data/simple_split/tasks_test_simple.txt", input_language=train_data.input_language, output_language=train_data.output_language)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=CollateFunctor())
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=CollateFunctor())

    logging.info(f"TRAIN: Number of words in input language: {train_data.input_language.n_words}")
    logging.info(f"TRAIN: Number of words in output language: {train_data.output_language.n_words}")
    logging.info(f"TRAIN: {train_data.output_language.word2index}")
    logging.info(f"TRAIN: {train_data.output_language.index2word}")

    logging.info(f"VAL: Number of words in input language: {val_data.input_language.n_words}")
    logging.info(f"VAL: Number of words in output language: {val_data.output_language.n_words}")
    logging.info(f"VAL: {val_data.output_language.word2index}")
    logging.info(f"VAL: {val_data.output_language.index2word}")


    model = MyTransformer(num_encoder_layers=3, num_decoder_layers=3, hidden_size=128, num_heads=4, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    device_max_steps = EPOCHS * len(train_loader)
    warmup_proportion = 0.6

    scheduler = cosine_schedule_with_warmup(optimizer, int(device_max_steps * warmup_proportion), device_max_steps, 0.1)

    params = sum([p.numel() for p in model.parameters()])
    logging.info(f"Model parameter count: {params}")
    
    for epoch in range(EPOCHS):
        train_loss = 0.0
        val_loss = 0.0
        train_corrects = 0.0
        train_n = 0
        val_corrects = 0.0
        val_n = 0
        model.train()
        for source_ids, source_mask, target_ids, target_mask, source_str, target_str in tqdm(train_loader):
            source_ids = source_ids.to(device)
            source_mask = source_mask.to(device)
            target_ids = target_ids.to(device)
            target_mask = target_mask.to(device)
            optimizer.zero_grad()
            out = model(source_ids, source_mask, target_ids[:,:-1], target_mask[:,:-1])
            loss = criterion(out.transpose(-2, -1), target_ids[:, 1:])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                preds = torch.argmax(out, dim=-1)
                correct = torch.sum(preds == target_ids[:, 1:])
                train_corrects += correct
                train_n += target_ids.shape[1]
        
        model.eval()
        with torch.no_grad():
            for source_ids, source_mask, target_ids, target_mask, source_str, target_str in tqdm(val_loader):
                source_ids = source_ids.to(device)
                source_mask = source_mask.to(device)
                target_ids = target_ids.to(device)
                target_mask = target_mask.to(device)
                out = model(source_ids, source_mask, target_ids[:,:-1], target_mask[:,:-1])
                loss = criterion(out.transpose(-2, -1), target_ids[:, 1:])
                val_loss += loss.item()
                preds = torch.argmax(out, dim=-1)
                correct = torch.sum(preds == target_ids[:, 1:])
                val_corrects += correct
                val_n += target_ids.shape[1]
            
            out = model(source_ids, source_mask, target_ids[:,:-1], target_mask[:,:-1])
            print([train_data.output_language.index2word[p.item()] for p in preds[0]])
            print([train_data.output_language.index2word[p.item()] for p in target_ids[0]])
        


        epoch_train_loss = train_loss / len(train_loader)
        train_accuracy = train_corrects/train_n
        epoch_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects/val_n
        logging.info(f"Epoch: {epoch}, train loss: {epoch_train_loss}, train accuracy: {train_accuracy}, val loss: {epoch_val_loss}, val accuracy: {val_accuracy}")