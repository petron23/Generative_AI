import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import build_transformer
from dataset import BilingualDataset, causal_mask

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokinizer(config, ds, lang):
    tokenizer_path = Path(config["tokinizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token= "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train")

    # Build tokenizer
    tokenizer_src = get_or_build_tokinizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokinizer(config, ds_raw, config["lang_tgt"])

    # 90-10 split
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(ds=train_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt,
                                 src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])
    val_ds = BilingualDataset(ds=val_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt,
                                 src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])
    
    max_len_src, max_len_tgt = 0, 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of the source sentence: {max_len_src}")
    print(f"max length of the target sentence: {max_len_tgt}")
        
    train_dataloader = DataLoader(dataset=train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len, src_seq_len=config["seq_len"], tgt_seq_len=config["seq_len"], d_model=config["d_model"])
    return 

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device is {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config=config)
    model = get_model(config=config, vocab_src_len=src_tokenizer.get_vocab_size(), vocab_tgt_len=tgt_tokenizer.get_vocab_size()).to(device)
    
    # Logging via Tensorboard
    