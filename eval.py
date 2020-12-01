# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import sys
import transformers
import torch

try:
    from src.tm_metric import compute_metric
except:
    # should the bashlexer or metric not be properly installed, print a warning.
    print("[WARNING]: Importing of metric computation failed. This may somewhat decrease performance.")
    # dummy function
    def compute_metric(*args, **kwargs):
        return 0



def predict_full(invocation, model1, model2, tokenizer1, tokenizer2, result_cnt=5):
    beam_count = 2 * result_cnt
    p1 = predict_single(model1, tokenizer1, invocation, top=beam_count)
    p2 = predict_single(model2, tokenizer2, invocation, top=beam_count)

    p1, p2 = set(p1), set(p2)
    if len(p1 | p2) == 0:
        print("Prediction failed!")
        prediction = ["date"]*result_cnt
        confidence = [0.]*result_cnt
        return prediction, confidence

    # start out with items that are both in p1 and p2
    # or else a maximally similar pair if there's not overlap
    if len(p1&p2) == 0:
        ins = {max(p1|p2, key=lambda x: word_simil(p1|p2, x))}
    else:
        ins = p1 & p2
            
    # remove interchangable items
    fpairs = get_equal_pairs(ins)
    fpairs = {frozenset(v) for k,v in fpairs.items()}
    for pair in fpairs:
        best = min(ins, key=lambda x: word_simil(ins, x))
        for pp in fpairs - {best}:
            ins.discard(pp)
            
    # in case of too much agreement,
    # remove most similar commands
    while len(ins) > result_cnt -1:
        best = max(ins, key=lambda x: word_simil(ins, x))
        ins.remove(best)
                
    # in case of not enough agreement,
    # add least similar commands
    overig = (p1 | p2) - ins
    fpairs = get_equal_pairs(overig)
    while len(ins) < result_cnt and len(overig) > 0:
        # not enough agreement: add diverse commands
        best = min(overig, key=lambda x: word_simil(ins, x))
        ins.add(best)
        overig.remove(best)
        for pp in fpairs[best]:
            overig.discard(pp)
        
    prediction = list(ins)
    confidence = [1.]*len(prediction)
    while len(prediction) < result_cnt:
        prediction.append('date')
        confidence.append(0.)

    return prediction, confidence
    

def get_tokenizer(model_name):
    return transformers.GPT2Tokenizer.from_pretrained(f'{model_name}')

def get_model(model_name):
    return transformers.GPT2LMHeadModel.from_pretrained(f'{model_name}')


def word_simil(refr, new):
    """ maximum word similarity between sentence and list of sentences """
    new = set(new.split(' '))
    refr = [set(x.split(' ')) for x in refr if x!=new]
    refr = [len(x&new) for x in refr]
    return max(refr)


# parameters to the template metric
SCORE_PARAMS = {'u1': 1., 'u2':1.}

def get_equal_pairs(combi):
    combi = list(combi)
    fpairs = dict()
    for p in combi:
        fpairs[p] = {p}
    for i in range(len(combi)):
        pi = combi[i]
        for j in range(i+1, len(combi)):
            pj = combi[j]
            score = compute_metric(pi, 1, pj, SCORE_PARAMS)
            if score == 1:
                fpairs[pi] = {pj}
                fpairs[pj].add(pi)
    return fpairs


def tokenize_query(tokenizer, prompt):
    """ Prepare input """
    prompt = f"<|endoftext|> english: {prompt}\nbash:"
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    return encoded_prompt


def decode(tokenizer, v):
    text = tokenizer.decode(v, clean_up_tokenization_spaces=False)
    # remove query at the start
    text = text[text.find("bash:")+5:]
    # remove possible junk at the end
    end = text.find("\n")
    if end!=-1:
        text = text[:end]
    text = text.replace('<|endoftext|>', '')
    text = text.strip('\n ')
    return text


def predict_single(model, tokenizer, prompt, top=1):
    prompt = tokenize_query(tokenizer, prompt)

    try:
        outputs = generate_single(
            model, tokenizer, prompt['input_ids'], 
            num_beams=top,
            num_return_sequences=top
        )
    except:
        print("WARNING: prediction failed")
        return []

    outputs = [decode(tokenizer, v) for v in outputs]
    return outputs


def generate_single(model, tokenizer, input_ids, num_beams=2, num_return_sequences=1):
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=200, # max_length less relevant as we do early stopping
        num_beams=num_beams,
        do_sample=False, # greedy
        num_return_sequences=num_return_sequences,
        eos_token_id=198, # halt on newline
        pad_token_id=tokenizer.eos_token_id
        )
    return output_sequences





def main():
    print('-- Loading models...')
    tokenizer1 = get_tokenizer('gpt2-large')
    model1 = get_model('jaron-maene/gpt2-large-nl2bash')

    tokenizer2 = get_tokenizer('gpt2-medium')
    model2 = get_model('jaron-maene/gpt2-medium-nl2bash')
    print('-- Models loaded')

    if len(sys.argv) < 2:
        print("Please provide a file with the commands to be translated. Refer to the README for more information")
        return

    with open(sys.argv[1], 'r') as f:
        invocations = [x.strip() for x in f.readlines()]

    for invocation in invocations:
        print("-- Starting prediction")
        print("invocation:")
        print(f"\t\t{invocation}")
        predictions, confidences = predict_full(invocations, model1, model2, tokenizer1, tokenizer2)
        print("results:")
        for p in predictions:
            print(f"\t\t{p}")



if __name__ == '__main__':
    main()
