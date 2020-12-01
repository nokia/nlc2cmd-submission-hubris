# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

from src.config import cfg


def read_data(filename):
    with open(cfg('data_path')+filename, 'r') as handle:
        content = handle.readlines()
    return [x.strip() for x in content]


def save_data(filename, content):
    if type(content) == type(list):
        content = "\n".join(lines)
    with open(cfg('data_path')+filename, 'w+') as handle:
        handle.write(content)


def context(pair):
    if '' in pair:
        return ''
    enc = f"{encode(pair[0])} {pair[1]}"
    if cfg('encoding').endswith('LBL'):
        return f"{enc} {cfg('eos')}\n"
    else:
        return f"{enc}\n"


def encode(prompt):
    return f"{cfg('eos')} {cfg('sep1')} {prompt}\n{cfg('sep2')}"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def decode(tokenizer, v):
    text = tokenizer.decode(v, clean_up_tokenization_spaces=False)
    # remove query at the start
    start = text.find(cfg('sep2')) + len(cfg('sep2'))
    text = text[start:]
    # remove possible junk at the end
    end = text.find("\n")
    if end!=-1:
        text = text[:end]
    text = text.strip('\n ')
    return text


def decode_batch(tokenizer, vs):
    decoded = [decode(tokenizer, v) for v in vs]
    return decoded