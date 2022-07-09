import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

print("load models")
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")
print("torch.cuda.is_available()",torch.cuda.is_available())

if torch.cuda.is_available():
    model = model.to("cuda")

text = "私は、"
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    print("generating")
    output_ids = model.generate(
        token_ids.to(model.device),
        max_length=100,
        min_length=100,
        do_sample=True,
        top_k=500,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_word_ids=[[tokenizer.unk_token_id]]
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

# print(type(model)) # <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
# print(model.__dict__)
"""{'training': False, '_parameters': OrderedDict(), '_buffers': OrderedDict(), '_non_persistent_buffers_set': set(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict(), '_forward_pre_hooks': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_load_state_dict_pre_hooks': OrderedDict(), '_load_state_dict_post_hooks': OrderedDict(), '_modules': OrderedDict([('transformer', GPT2Model(), 'config': GPT2Config {
  "_name_or_path": "rinna/japanese-gpt-1b",
  "activation_function": "gelu_fast",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 2,
  "embd_pdrop": 0.1,
  "eos_token_id": 3,
  "gradient_checkpointing": false,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 2048,
  "n_head": 16,
  "n_inner": 8192,
  "n_layer": 24,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_use_proj": true,
  "transformers_version": "4.20.1",
  "vocab_size": 44928
}
, 'name_or_path': 'rinna/japanese-gpt-1b', 'warnings_issued': {}, 'model_parallel': False, 'device_map': None}
"""