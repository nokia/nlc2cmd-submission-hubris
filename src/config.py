# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import copy 

# default config
DEFAULT = {
	# data configuration
	'sep1': 'english:',           # prefix for nl utterance
	'sep2': 'bash:',              # prefix for bash command
	'encoding': 'blocked',        # options: 'LBL', 'blocked', 'text'
	'max_line': 170,
	'max_block': 400,
	'data_path': 'data/nl2bash/', # folder where data is saved
	'eos': '<|endoftext|>',       # end of sequence token

	# training params
	'model': 'gpt2',              # name of model, must be from gpt2 family
	'device': 'cuda',
	'epochs': 12,                 # total amount of epochs to train for
	'batch_size': 1,
	'grad_acc': 1,                # amount of gradient accumulation steps
	'out_path': 'output/nl2bash/',# folder where runs are saved

	# generation params (for validation)
	'val_metric': 'template',     # validation metric, 'BLEU' or 'template'
	'beams': 5,                   # beam width in beam search
	'val_n': 5,
	'max_gen': 200,               # max length of command (in tokens, early stopping on newline is used so actual value not that important)

	# varia
	'add_tokens': False,          # add new tokens to tokenizer for common bash utilities
	'random_init': False,         # start training from random intialized model instead of pretrained
}

def load_cfg(path):
	with open(path + '/conf.txt') as f:
		cfg = eval(f.read())
	set_cfg(cfg)

def set_cfg(k, v=None):
	global CFG
	if v!= None:
		CFG[k] = v
	else:
		assert type(k) == dict
		CFG = k

def cfg(q=None):
	if q!=None:
		return CFG[q]
	else:
		return copy.deepcopy(CFG)

def filename(mode, typ=""):
    if len(typ) == 0:
        return f"{cfg('data_path')}{mode}.txt"
    return f"{cfg('data_path')}{mode}_{typ}.txt"

def reset_cfg():
	global CFG
	CFG = copy.deepcopy(DEFAULT)

reset_cfg()

