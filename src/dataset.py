# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import torch
import os
import random
import math

from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from src.config import cfg, filename
from src.data_utils import read_data


class EmptyDataset(Dataset):
    def __init__(self):
        self.examples = []
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long, device=cfg('device'))

class CombinedDataset(EmptyDataset):
    def __init__(self, d1: Dataset, d2: Dataset):
        assert len(d1) < len(d2)
        self.d1 = d1
        self.d2 = d2
        self.x = 0
        
    def __len__(self):
        return 2*len(self.d1)

    def __getitem__(self, i) -> torch.Tensor:
        if i%2==0:
            return self.d1[i//2]
        else:
            self.x = (self.x+1) % len(self.d2)
            return self.d2[self.x]


class LBLDataset(EmptyDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, debug=False):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines()]

        pairs = [x+"\n"+y for x,y in zip(lines[::2],lines[1::2])]
        if debug:
            for i in pairs[:2]: print(i)
        batch_encoding = tokenizer(pairs, add_special_tokens=True, padding=True, truncation=True, max_length=cfg('max_line'))
        self.examples = batch_encoding["input_ids"]
        if debug:
            for i in self.examples[:2]: print(i)


class BlockedDataset(EmptyDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, debug=False):
        assert os.path.isfile(file_path)
        block_size = cfg('max_block')
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.examples = []
        print(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
            
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        eos = tokenizer.eos_token_id
        cblock = [tokenized_text.pop(0)]
        
        while tokenized_text:
            t = tokenized_text.index(eos)
            if t == -1:
                break
            if t > block_size:
                # if entry doesn't fit in block, throw it away
                tokenized_text = tokenized_text[t+1:]
                print(f"Throwing away {t} tokens.")
            elif t + len(cblock) <= block_size:
                cblock += tokenized_text[:t+1]
                tokenized_text = tokenized_text[t+1:]
            else:
                rest = block_size - len(cblock)
                
                cblock = cblock + [eos]*rest
                self.examples.append(cblock)
                cblock = [eos]
        
        if debug:
            for i in self.examples[:1]: 
                print(i)
                print(tokenizer.decode(i))

# todo set back to dev
def get_validation_data():
    dev_cm = read_data('dev_cm.txt')
    dev_nl = read_data('dev_nl.txt')

    dev_dict = {}
    for x, y in zip(dev_cm, dev_nl):
        if y not in dev_dict.keys():
            dev_dict[y] = [x]
        else:
            dev_dict[y].append(x)
        
    dev_dataset = list(zip(*dev_dict.items()))
    #dev_dataset = [x[:cfg('val_n')] for x in dev_dataset]
    return dev_dataset