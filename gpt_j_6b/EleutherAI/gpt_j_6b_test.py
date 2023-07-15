from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        torch_dtype=torch.float16,
    ).eval().cuda()

    tokenizer.save_pretrained("gpt_j_6b/models")
    model.save_pretrained("gpt_j_6b/models")

    prompt = ''
    while True:
        try:
            text = input('>')
            if text in ['exit']:
                break
        except:
            break

        prompt += text
        tokens = tokenizer(prompt, return_tensors='pt').input_ids
        token_length = len(tokens)
        generated_tokens = model.generate(
            tokens.long().cuda(),
            use_cache=True,
            do_sample=True,
            temperature=1,
            top_p=0.9,
            repetition_penalty=1.125,
            min_length=1,
            max_length=len(tokens[0]) + 400, pad_token_id=tokenizer.eos_token_id
        )
        last_tokens = generated_tokens[0][token_length:]
        generated_text = tokenizer.decode(last_tokens)
        print(generated_text)
        prompt += generated_text

if __name__ == '__main__':
    main()