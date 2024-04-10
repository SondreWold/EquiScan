import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import ScanData, CollateFunctor
from tqdm import tqdm
import math
import logging
import numpy as np
from enum import Enum
import random
from model import Transformer
import wandb
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for EquiScan training script")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--task", type=str, default="simple")
    parser.add_argument("--p", type=str, default="64")
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--log", action='store_true', help="Trigger WANDB logging")
    parser.add_argument("--deepset", action='store_true', help="Use DeepSet method")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyTransformer(nn.Module):
    def __init__(self, vocab_size_input=16, vocab_size_output=9, num_encoder_layers=3, num_decoder_layers=3, hidden_size=128, num_heads=4, dropout=0.1):
        super(MyTransformer, self).__init__()
        self.emb_in = nn.Embedding(vocab_size_input, hidden_size)
        self.emb_out = nn.Embedding(vocab_size_output, hidden_size)
        self.transformer_model = Transformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.projection = nn.Linear(hidden_size, vocab_size_output)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.emb_out.weight.data.uniform_(-initrange, initrange)
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.emb_in(src)
        tgt = self.emb_out(tgt)
        out = self.transformer_model(src, src_mask, tgt, tgt_mask)
        out = self.projection(out)
        return out

    def encode_source(self, source, source_padding_mask):
        encoded = self.emb_in(source)
        return self.transformer_model.encode_source(encoded, source_padding_mask)

    def decode_step(self, memory, source_padding_mask, target, target_padding_mask):
        encoded = self.emb_out(target)
        decoded = self.transformer_model.decoder(encoded, target_padding_mask, memory, source_padding_mask)
        return self.projection(decoded)


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)
    logging.info("-----EquiScan------")

    wandb_config = vars(args).copy()
    del wandb_config["log"]
    logging.info(f"Running EquivScan with config: {wandb_config}")

    if args.task == "simple":
        train_path = f"./data/simple_split/size_variations/tasks_train_simple_p{args.p}.txt"
        val_path = f"./data/simple_split/size_variations/tasks_test_simple_p{args.p}.txt"
    elif args.task == "length":
        del wandb_config["p"]
        train_path = f"./data/length_split/tasks_train_length.txt"
        val_path = f"./data/length_split/tasks_test_length.txt"

    if args.log:
        wandb.init(project="equiscan", config=wandb_config, entity="sondrewo")
    train_data = ScanData(train_path)
    val_data = ScanData(val_path, input_language=train_data.input_language, output_language=train_data.output_language)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=CollateFunctor(), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, collate_fn=CollateFunctor())
    EPOCHS = 5
    device_max_steps = EPOCHS * len(train_loader)

    logging.info(f"TRAIN: Number of words in input language: {train_data.input_language.n_words}")
    logging.info(f"TRAIN: Number of words in output language: {train_data.output_language.n_words}")
    logging.info(f"VAL: Number of words in input language: {val_data.input_language.n_words}")
    logging.info(f"VAL: Number of words in output language: {val_data.output_language.n_words}")

    model = MyTransformer(num_encoder_layers=args.layers, num_decoder_layers=args.layers, hidden_size=args.hidden_size, num_heads=args.heads, dropout=args.dropout).to(device)
    params = sum([p.numel() for p in model.parameters()])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = cosine_schedule_with_warmup(optimizer, int(device_max_steps * args.warmup_proportion), device_max_steps, 0.1)

    logging.info(f"Model parameter count: {params}")
    logging.info(f"Total steps: {device_max_steps}")
    logging.info(f"Total epochs: {EPOCHS}")
    for epoch in range(EPOCHS):
        train_loss = 0.0
        val_loss = 0.0
        train_corrects = 0.0
        train_n = 0
        val_corrects = 0.0
        val_n = 0
        model.train()
        for source_ids, source_mask, target_ids, target_mask, source_str, target_str in tqdm(train_loader):
            optimizer.zero_grad()
            B, T = source_ids.shape
            source_ids = source_ids.to(device)
            source_mask = source_mask.to(device)
            target_ids = target_ids.to(device)
            target_mask = target_mask.to(device)
            out = model(source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1])
            loss = criterion(out.transpose(-2, -1), target_ids[:, 1:])
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.log:
                wandb.log({"learning_rate_decay": optimizer.param_groups[0]["lr"], "train_loss": loss.item()})
            with torch.no_grad():
                preds = torch.argmax(out, dim=-1)
                gold = target_ids[:, 1:].flatten()
                preds = preds.flatten()
                preds = preds[gold != 0]
                gold = gold[gold != 0]
                correct = torch.sum(preds == gold)
                train_corrects += correct
                train_n += len(gold)
        model.eval()
        with torch.no_grad():
            for source_ids, source_mask, target_ids, target_mask, source_str, target_str in tqdm(val_loader):
                B, T = source_ids.shape
                source_ids = source_ids.to(device)
                source_mask = source_mask.to(device)
                target_ids = target_ids.to(device)
                target_mask = target_mask.to(device)
                out = model(source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1])
                loss = criterion(out.transpose(-2, -1), target_ids[:, 1:])
                val_loss += loss.item()
                preds = torch.argmax(out, dim=-1)
                gold = target_ids[:, 1:].flatten()
                preds = preds.flatten()
                preds = preds[gold != 0]
                gold = gold[gold != 0]
                correct = torch.sum(preds == gold)
                val_corrects += correct
                val_n += len(gold)
        epoch_train_loss = train_loss / len(train_loader)
        train_accuracy = train_corrects/train_n
        epoch_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects/val_n
        logging.info(f"Epoch: {epoch}, train loss: {epoch_train_loss}, train accuracy: {train_accuracy}, val loss: {epoch_val_loss}, val accuracy: {val_accuracy}")
        if args.log:
            wandb.log({"val_loss": epoch_val_loss, "val_acc": val_accuracy, "train_acc": train_accuracy})
