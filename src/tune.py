# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

from src.run import main
from src.config import set_cfg, reset_cfg


def experiments():
	for i in range(1):
		reset_cfg()
		set_cfg('encoding', 'inter_blocked')
		set_cfg('model', 'gpt2-large')
		set_cfg('epochs', 10)
		set_cfg('data_path', 'data/clai/')
		set_cfg('out_path', 'output/clai2/')
		set_cfg('val_metric', 'template')
		set_cfg('max_block', 200)
		set_cfg('batch_size', 2)
		set_cfg('grad_acc', 16)
		set_cfg('device', 'cpu')
		main()


if __name__ == '__main__':
	experiments()
