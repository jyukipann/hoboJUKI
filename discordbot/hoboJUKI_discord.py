import discord
import hoboJUKI_IDs
import re
from collections import deque
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM


class HoboJUKI(discord.Client):
    def __init__(self, *, intents, **options) -> None:
        super().__init__(intents=intents, **options)
        print("loading models")
        self.mention_pattern = None
        self.mention_user_pattern = re.compile(f'<@\d+>')
        self.last_message_queue = deque([], maxlen=5)
        self.reply_sub_words = re.compile("<s>|</s>|[UNK]|<unk>")
        self.s_tag = re.compile("<s>|</s>")
        self.target_channel_id = 790812986599931967
        # self.target_channel_id = 1004647945435623454 #テストチャンネル

        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
        self.model = AutoModelForCausalLM.from_pretrained("hoboJUKI_model/")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def generate_reply(self, input_message):
        input_message = self.mention_user_pattern.sub("", input_message)
        input_message += "[REP][ほぼじゅき]<s>"
        
        token_ids = self.tokenizer.encode(
            input_message, add_special_tokens=False, return_tensors="pt")
        input_message_count = len(token_ids[0])
        with torch.no_grad():
            # print("generating")
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_length=1000,
                min_length=15,
                do_sample=True,
                top_k=500,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_word_ids=[[self.tokenizer.unk_token_id]],
                num_beams=5,
                # early_stopping=True,
            )
        reply = self.tokenizer.decode(output_ids.tolist()[0][input_message_count:])
        input_message = self.s_tag.sub(" ",input_message)
        reply = reply.replace(input_message,"")
        reply = self.reply_sub_words.sub("", reply)
        return reply

    async def on_ready(self):
        self.mention_pattern = re.compile(f'<@{self.user.id}>')
        print('Logged on as', self.user)

    async def on_message(self, message):
        if message.channel.id != self.target_channel_id:
            return
        author_nick_name = message.author.nick
        if author_nick_name is None:
            author_nick_name = message.author.name
        message_content = message.content
        # don't respond to ourselves
        if message.author == self.user:
            self.last_message_queue.append(
                f"[REP][ほぼじゅき]<s>{message_content}</s>")
            return
        else:
            self.last_message_queue.append(
                f"[{author_nick_name}]<s>{message_content}</s>")
        if self.mention_pattern.search(message.content) is not None:
            # print("meintined")
            reply_message = None
            reply_message = self.generate_reply(
                "".join(self.last_message_queue))
            if reply_message is not None:
                await message.channel.send(reply_message)


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = HoboJUKI(intents=intents)
    client.run(hoboJUKI_IDs.token)
