import rinna_utils
import torch

def load_tokenizer():
    tokenizer = rinna_utils.load_tokenizer('./hoboJUKI-3.6b/models/rinna3.6b')
    return tokenizer

def load_model():
    model = rinna_utils.load_model('./hoboJUKI-3.6b/models/rinna3.6b')
    model = rinna_utils.load_peft_model(model, 'hoboJUKI-3.6b/models/lora-rinna-3.6b')
    return model

generate = rinna_utils.generate

model = load_model()
model.eval()

print(generate('ツイート:<s>', load_tokenizer(), model))