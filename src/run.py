# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import random
import os
from datetime import datetime
import numpy as np

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments, BertForMaskedLM
)

from src.config import cfg, filename
from src.generate import validate
from src.dataset import LBLDataset, BlockedDataset, CombinedDataset, get_validation_data
from src.trainer import MTrainer
from src.preprocess import preprocess

import bashlint.bash as bashinfo


def get_trainer(train_dataset, collator, model):
    training_args = TrainingArguments(
        output_dir = f'output/bash',
        overwrite_output_dir = True,
        do_train = True,
        no_cuda = cfg('device')=='cpu',
        num_train_epochs = cfg('epochs'),
        per_device_train_batch_size = cfg('batch_size'),
        gradient_accumulation_steps = cfg('grad_acc'),
        logging_steps = 5,
        save_steps = 0,
        seed = random.randint(0,2**32-1))
    trainer = MTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        prediction_loss_only = True)
    return trainer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(cfg('model'))
    # gpt2 has no padding by default
    try:
        if cfg('eos') != tokenizer.eos_token:
            print(f"Warning: non-default eos token (default is {tokenizer.eos_token})")
    except:
        print("Warning: no default eos token")
    tokenizer.add_tokens(cfg('eos'))
    tokenizer.eos_token = cfg('eos')
    print("EOS", tokenizer.eos_token, tokenizer.eos_token_id)
    # add eos as pad_token should there be no pad token
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    print("PAD", tokenizer.pad_token, tokenizer.pad_token_id)
    try:
        if cfg('add_tokens'):
            # assert that there are tokens for seperators and common bash tools
            #added = tokenizer.add_tokens([cfg('sep1'), cfg('sep2')])
            added = tokenizer.add_tokens(bashinfo.top_100_utilities)
            print(f"added {added} tokens")
    except:
        pass
    return tokenizer


def get_model(tokenizer, resume=False):
    if cfg('random_init'):
        # load randomly initialized model instead of pretrained
        model_config = transformers.GPT2Config()
        model = transformers.GPT2LMHeadModel(model_config)
    elif resume:
        # resume from previous best
        model = AutoModelForCausalLM.from_pretrained(cfg('out_path') + cfg('name'))
    else:
        # load pretrained model
        model = AutoModelForCausalLM.from_pretrained(cfg('model'))
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(cfg('device'))
    return model


def get_session_path():
    path = cfg('out_path') + datetime.now().strftime("%m-%d_%H:%M:%S") + '/'
    os.mkdir(path)
    return path


def save(path, content):
    with open(path, 'a+') as f:
        f.write(content)


BEST_metric = 0

def main():
    print("PREPROCESSING DATA")
    preprocess()
    print("LOADING TOKENIZER")
    tokenizer = get_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("LOADING MODEL", cfg('model'))
    model = get_model(tokenizer)

    print("LOADING DATA")
    if cfg('encoding') == 'LBL':
        train_dataset = LBLDataset(tokenizer=tokenizer, file_path=filename('train'))
    elif cfg('encoding') == 'blocked':
        train_dataset = BlockedDataset(tokenizer=tokenizer, file_path=filename('train'))
    elif cfg('encoding') == 'text':
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=filename('train'), block_size=cfg('max_block'))
    elif cfg('encoding').startswith('inter'):
        if cfg('encoding').endswith('LBL'):
            loader = LBLDataset
        elif cfg('encoding').endswith('blocked'):
            loader = BlockedDataset

        d1 = loader(tokenizer=tokenizer, file_path=filename('train'))
        d2 = loader(tokenizer=tokenizer, file_path=filename('dirty'))
        train_dataset = CombinedDataset(d1, d2)
    else:
        raise ValueError("Unkown encoding")

    trainer = get_trainer(train_dataset, data_collator, model)

    def validator(x, y):
        global BEST_metric
        model.save_pretrained(session)
        metric, pred = validate(model, tokenizer, x, y)
        if np.mean(metric) > BEST_metric:
            print("NEW BEST (saving)")
            BEST_metric = np.mean(metric)

        # save predicitions and model
        save(session+"metric.txt", str(metric)+"\n")
        save(session+"pred.txt", str(pred)+"\n\n")
        return metric, pred

    trainer.validator = validator
    trainer.val_dataset = get_validation_data()

    # saving configuration
    print("SAVING...")
    session = get_session_path()
    print(session)
    save(session+"conf.txt", repr(cfg()))

    print("STARTING TRAINING...")
    trainer.train()

if __name__ == '__main__':
    main()