# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import numpy as np
import transformers
import pathlib

from transformers.convert_graph_to_onnx import convert
from onnxruntime_tools import optimizer
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

from src.data_utils import encode, decode
from src.run import get_tokenizer
from src.config import load_cfg, cfg


def convert_to_onnx(model_path, model_name, output_path):
    output_path = pathlib.Path(output_path)
    convert(framework="pt", model=model_path, tokenizer=model_name, output=output_path, opset=11)
    if model_name in ('gpt2',): # gpt2-medium not supported yet
        optimized_model = optimizer.optimize_model(output_path, model_type=model_name)
        optimized_model.save_model_to_file(output_path)

def prepare_onnx_generation(model_path, device):
    load_cfg(model_path)
    print("# Converting model")
    convert_to_onnx(model_path, cfg('model'), 'onnx/model.onnx')
    print("# Loading model into huggingface")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    if device == 'cuda':
        provider = 'CUDAExecutionProvider'
    else:
        raise ValueError("Unkown device")
    print("# Loading model into ONNX")
    onnx_model = create_model_for_provider('onnx/model.onnx', provider)
    print("# Loading tokenizer")
    tokenizer = get_tokenizer()
    return onnx_model, model.lm_head, tokenizer

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    # Few properties than might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    # Load the model as a graph and prepare the CPU backend 
    return InferenceSession(model_path, options, providers=[provider])

def onnx_predict(onnx_model, lm_head, tokenizer, query, device='cuda'):
    query = encode(query)
    model_inputs = tokenizer.encode_plus(query, return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items() if k=='input_ids'}

    # Run the model (None = get all the outputs)
    tprob = []
    for _ in range(100):
        #print(inputs_onnx)
        logits = onnx_model.run(None, inputs_onnx)[0][:,-1,:]
        logits = lm_head(torch.tensor(logits).to(device))
        log_probs = F.softmax(logits, dim=-1)
        prob, prev = torch.topk(log_probs, k=1)
        tprob.append(prob)
        token = prev.item()
        if token == 198:
            break
        inputs_onnx['input_ids'] = np.atleast_2d(np.append(inputs_onnx['input_ids'], prev.cpu().numpy()))

    tprob = np.mean([x.item() for x in tprob])
    output = decode(tokenizer, inputs_onnx['input_ids'][0]) 
    
    return output , tprob