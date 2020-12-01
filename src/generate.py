# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from transformers import PreTrainedModel

from src.config import cfg
from src.data_utils import encode, decode_batch, chunks
import src.modified_beam_search as mbs 
from src.tm_metric import compute_metric


def flatten(l):
    if len(l) == 1:
        return l[0]
    return l

def tokenize_query(tokenizer, prompt, device):
    """ Prepare input """
    prompt = encode(prompt)
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
    return encoded_prompt.to(device)


def predict_single(model, tokenizer, prompt, top=1, device=None, max_length=None, beams=None):
    if device == None:
        device = cfg('device')
    if max_length == None:
        max_length = cfg('max_gen')
    if beams == None:
        beams = cfg('beams')

    prompt = tokenize_query(tokenizer, prompt, device)
    model = model.to(device)

    output_sequences = model.generate(
        input_ids=prompt,
        max_length=max_length,
        num_beams=max(top, beams),
        do_sample=False,
        num_return_sequences=top,
        pad_token_id=tokenizer.eos_token_id
        )
    output = decode_batch(tokenizer, output_sequences)
    return output


def predict_single_mod(model, tokenizer, prompt, top=1, device=None, beams=None):
    if device == None:
        device = cfg('device')
    if beams == None:
        beams = cfg('beams')
    prompt = tokenize_query(tokenizer, prompt, device)

    # bit hacky
    PreTrainedModel._generate_beam_search = mbs._generate_beam_search
    output_sequences, output_scores = model.generate(
        input_ids=prompt,
        max_length=300, # max_length less relevant as mod does early stopping
        num_beams=max(top, beams),
        do_sample=False,
        num_return_sequences=top,
        pad_token_id=tokenizer.eos_token_id
        )
    output = decode_batch(tokenizer, output_sequences)
    return output, output_scores

    
def predict_diverse(model, tokenizer, prompt, temp, top_p, top=1):
    prompt = tokenize_query(tokenizer, prompt)

    output_sequences = model.generate(
        input_ids=prompt,
        max_length=cfg('max_gen'),
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=top,
        pad_token_id=tokenizer.eos_token_id
        )
    output = decode_batch(tokenizer, output_sequences)
    if len(output) == 1:
        return output[0]
    else:
        return output


def predict_batch(model, tokenizer, input_ids, top=1):
    """ Requires all inputs to be of equal length """
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.to(cfg('device'))

    # prediction
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=cfg('max_gen'),
        num_beams=cfg('beams'),
        do_sample=False,
        num_return_sequences=top,
        pad_token_id=tokenizer.eos_token_id
        )
    print(output_sequences.shape)
    output = decode_batch(tokenizer, output_sequences)
    return output



    
def predict_all(model, tokenizer, prompts, top=1):
    prompts = [encode(prompt) for prompt in prompts]
    enc = tokenizer(prompts, padding=False, truncation=False)['input_ids']
    CHUNK_SIZE = 4

    # group by length 
    d = {}
    for i, x in enumerate(enc):
        length = len(x)
        if length in d.keys():
            d[length].append((x,i))
        else:
            d[length] = [(x,i)]

    pred_cm = [""]*len(prompts)
    for k, v in tqdm(d.items()):
        pr, ind = zip(*v)
        r = []
        for chunk in chunks(tuple(pr), CHUNK_SIZE):
            r += predict_batch(model, tokenizer, chunk, top)
        for i, x in zip(ind, r):
            pred_cm[i] = x
    return pred_cm


def get_template_score(dev_cm, pred_cms):
    PARAMS = {'u1': 1., 'u2':1.}
    scores = [compute_metric(pred_cm, 1, dev_cm, PARAMS)
                for pred_cm in pred_cms]

    if any(x > 0 for x in scores):
        total_score = max(scores)
    else:
        total_score = sum(scores) / len(scores)
    return total_score


def validate(model, tokenizer, dev_nls, dev_cms):
    outputs = [predict_single_mod(model, tokenizer, dev_nl, top=cfg('val_n')) 
                for dev_nl in dev_nls]
    
    confidences = [x[1] for x in outputs]
    predictions = [x[0] for x in outputs]

    scores_template = [get_template_score(dev_cm[0], pred_cm) 
                for (pred_cm, dev_cm) in zip(predictions, dev_cms)]
    print(f"[DEBUG]: TM score {np.mean(scores_template)}")

    scores_blue = [sentence_bleu(dev_cm, pred_cm[0]) 
                for (pred_cm, dev_cm) in zip(predictions, dev_cms)]
    print(f"[DEBUG]: BLUE score {np.mean(scores_blue)}")

    if cfg('val_metric') == 'BLUE':
        scores = scores_blue
    elif cfg('val_metric') == 'template':
        scores = scores_template
    else:
        assert False, f"Unkown validation metric '{metric}'"
    return scores, predictions