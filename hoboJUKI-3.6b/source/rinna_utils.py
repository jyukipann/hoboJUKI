"""
cd /work/tanimoto.j/workspace/GitHub/llm; conda activate llm
python rinna/train_lora.py
"""
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Tuple
from numpy import ndarray
from peft import PeftModel

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer


def load_model(model_name: str, quantization_config = None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        load_in_4bit=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available() and False:
        model = model.to("cuda")
    return model


def load_peft_model(model, peft_name: str):
    model = PeftModel.from_pretrained(
        model,
        peft_name,
        device_map="auto",
    )
    if torch.cuda.is_available() and False:
        model = model.to("cuda")
    return model


def tokenize(prompt: str, tokenizer, max_length=2048) -> Dict[str, ndarray]:
    result = tokenizer(
        prompt,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


def load_dataset(path: str) -> List[str]:
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    return prompts


def train_val_split(prompts: List[str], split_ratio: float = 0.8) -> Tuple[List[str]]:
    split_len = int(len(prompts)*split_ratio)
    train_prompts, val_prompts = prompts[:split_len], prompts[split_len:]
    return train_prompts, val_prompts


def tokenize_prompts(prompts: List[str], tokenizer) -> List[Dict[str, ndarray]]:
    return [tokenize(prompt, tokenizer) for prompt in prompts]


def get_lora_config():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config


def set_model_for_lora_training(model, lora_config):
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    return model


def generate(prompt, tokenizer, model, max_new_tokens=150, output_only_generated=True, escape_special_token=True):
    input_ids = tokenize(prompt, tokenizer)['input_ids'].cuda()
    input_ids = input_ids[:, :-1]
    input_token_len = input_ids.shape[1]
    # print(input_ids)
    # print(input_token_len)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = outputs[0]
    if output_only_generated:
        outputs = outputs[input_token_len:]
    if escape_special_token:
        outputs = outputs[outputs != 2]
        outputs = outputs[outputs != 3]
    text: str = tokenizer.decode(outputs.tolist())

    if escape_special_token:
        text = text.replace("<NL>", '\n')

    return text
