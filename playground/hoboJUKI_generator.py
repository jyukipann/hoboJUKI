import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

# python playground/hoboJUKI_generator.py

print("load models")
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
tokenizer.add_tokens(["[TWT]","[REP]", "[UNK]"])
model = AutoModelForCausalLM.from_pretrained("output_v1/")
model.resize_token_embeddings(len(tokenizer))
# print("torch.cuda.is_available()",torch.cuda.is_available())

if torch.cuda.is_available():
    model = model.to("cuda")

while True:
    text = input(">")
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        print("generating")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_length=100,
            min_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_word_ids=[[tokenizer.unk_token_id]],
            num_beams=5, 
            early_stopping=True,
        )
    output = tokenizer.decode(output_ids.tolist()[0])
    print(output)