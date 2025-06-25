import os
import torch
from typing import List, Optional
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, \
                        LlamaConfig, LlamaTokenizer, LlamaModel, \
                        AutoConfig, AutoTokenizer, AutoModelForCausalLM
from openprompt.plms import LMTokenizerWrapper
from collections import namedtuple
import transformers

from config import cfg
from lora import peft_model


ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))
_MODEL_CLASSES = {
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2Model,
        'wrapper': LMTokenizerWrapper
    }),
    "qwen2": ModelClass(**{
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForCausalLM,
        "wrapper": LMTokenizerWrapper
    }),    
    "qwen2.5": ModelClass(**{
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForCausalLM,
        "wrapper": LMTokenizerWrapper
    }),
    "llama": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel,
        "wrapper": LMTokenizerWrapper
    }),
}



def load_plm(model_name, model_path, specials_to_add = None, **kwargs):

    # print("model_name: ",model_name)
    # print("_MODEL_CLASSES: ",_MODEL_CLASSES)
    model_class = _MODEL_CLASSES[model_name]
    print("model_class: ",model_class)
    print("modelclass.config: ",model_class.config)
    # model_config = model_class.config.from_pretrained(model_path)
    # model = model_class.model.from_pretrained(model_path, config=model_config)
    # model_config = model_class.config.from_pretrained(model_path)
    model = model_class.model.from_pretrained(model_path)

    tokenizer = model_class.tokenizer.from_pretrained(model_path) 
    
    wrapper = model_class.wrapper
    
    if specials_to_add is None:
        specials_to_add = ['<pad>']
    else:
        specials_to_add.append('<pad>')

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    
    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, None, wrapper


def add_special_tokens(model: transformers.modeling_utils.PreTrainedModel,
                       tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):

    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                print("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer


def get_plm(
        plm_model_name: str,
        plm_model_path: str,
        plm_model_size: str,
        lora_rank: int,
        device: torch.device,
):
    print('plm_model_name:',plm_model_name)
    print('plm_model_path:',plm_model_path)
    print('plm_model_size:',plm_model_size)
    plm, *_ = load_plm(plm_model_name, plm_model_path,)
    plm = plm.to(device)    
    plm = peft_model(plm, plm_model_name, lora_rank)
    plm_embed_size = cfg.plm_embed_sizes[plm_model_name][plm_model_size]

    return plm, plm_embed_size