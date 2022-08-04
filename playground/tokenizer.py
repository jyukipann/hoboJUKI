import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
tokenizer.add_tokens(["[TWT]","[REP]", "[UNK]"])
# tokenizer.add_special_tokens({"unk_token":"[UNK]"})

text = "[CLS]<s>[SEP]</s>[TWT][REP]"
text = "[REP][TWT][UNK]<unk>"
token_ids = tokenizer.encode(text,add_special_tokens=True, return_tensors="pt")
print(len(token_ids[0]))
# token_ids = tokenizer(text,add_special_tokens=True, return_tensors="pt")
output = tokenizer.decode(token_ids.tolist()[0])
print(output)
print(token_ids.tolist()[0])
print(tokenizer.pad_token_id)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.sep_token_id)
print(tokenizer.cls_token_id)
print(tokenizer.unk_token_id)
print(tokenizer.__dict__)
print(tokenizer.special_tokens_map)