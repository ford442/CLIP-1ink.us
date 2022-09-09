import hashlib
import os
import urllib
import warnings
from typing import Any,Union,List
from pkg_resources import packaging
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from tqdm import tqdm
from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from functools import lru_cache as cache;
from methodtools import lru_cache as class_cache;

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC=InterpolationMode.BICUBIC
except ImportError:
    BICUBIC=Image.BICUBIC
if packaging.version.parse(torch.__version__)<packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")
__all__=["available_models","load","tokenize"]
_tokenizer=_Tokenizer()

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
    ])
def available_models() -> List[str]:
    return list(_MODELS.keys())
def load(fp16bit,sIze,name):
    device=torch.device("cuda:0")
    jit=False
    model_path=name
    with open(model_path, 'rb') as opened_file:
        state_dict=torch.load(opened_file,map_location=torch.device("cpu"))
        model=build_model(fp16bit,state_dict or model.state_dict())
        return model,_transform(sIze)
    return model,_transform(sIze)
    if fp16bit==False:
        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)
    return model,_transform(sIze)

#def tokenize(texts:Union[str,List[str]],context_length:int=77,truncate:bool=False)->Union[torch.IntTensor,torch.LongTensor]:
def tokenize(texts:Union[str,List[str]],context_length:int=77,truncate:bool=False)->Union[torch.ShortTensor]:
    if isinstance(texts,str):
        texts=[texts]
    sot_token=_tokenizer.encoder["<|startoftext|>"]
    eot_token=_tokenizer.encoder["<|endoftext|>"]
    all_tokens=[[sot_token]+_tokenizer.encode(text)+[eot_token]for text in texts]
    #if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
    #    result=torch.zeros(len(all_tokens),context_length,dtype=torch.long)
    #else:
    result=torch.zeros(len(all_tokens),context_length,dtype=torch.int16)
    for i, tokens in enumerate(all_tokens):
        if len(tokens)>context_length:
            if truncate:
                tokens=tokens[:context_length]
                tokens[-1]=eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i,:len(tokens)]=torch.tensor(tokens)
    return result
