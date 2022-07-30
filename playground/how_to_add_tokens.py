# https://github.com/huggingface/transformers/issues/6789

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(['<some_token_1>', '<some_token_2'>])
# model.resize_token_embeddings(len(tokenizer))