# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

from src.config import cfg
from src.data_utils import context, read_data, save_data

def preprocess():
	for NAME in ('dev', 'train', 'dirty'):
		try:
			cm = read_data(NAME+'_cm.txt')
			nl = read_data(NAME+'_nl.txt')
		except:
			print(f"[WARNING]: {NAME} data not found")
			continue

		al = [context(x) for x in zip(nl, cm)]
		al = al = "".join(al)
		if not al.endswith(cfg('eos')):
			al += cfg('eos')

		save_data(NAME+".txt", al)


if __name__ == '__main__':
	preprocess()