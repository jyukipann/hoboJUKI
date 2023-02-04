import discord
from discord.ext import tasks
import dmwebhook
import hoboJUKI_IDs
import re
from collections import deque
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
import time
import asyncio
import random

class HoboJUKI(discord.Client):
    def __init__(self, *, intents, **options) -> None:
        super().__init__(intents=intents, **options)
        print("loading models")
        self.mention_pattern = None
        self.mention_user_pattern = re.compile(f'<@\d+>')
        self.reply_sub_words = re.compile("<s>|</s>|[UNK]|<unk>|\[\]")
        self.s_tag = re.compile("<s>|</s>")
        self.target_channel_id = 790812986599931967
        # self.target_channel_id = 1004647945435623454 #テストチャンネル
        self.message_queue_max_length = 5
        self.last_message_queues = {self.target_channel_id:deque(maxlen=self.message_queue_max_length)}
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
        self.tokenizer.add_tokens(["[TWT]","[REP]", "[UNK]"])
        self.model = AutoModelForCausalLM.from_pretrained("hoboJUKI_model/")
        self.model.resize_token_embeddings(len(self.tokenizer))
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.dm_logger = None
        self.reply_sleep_time_range = [1,5]
        self.last_message_times = {}
        self.last_message_times[self.target_channel_id] = time.time()
        self.next_tweet_date = None

    def generate_tweet(self):
        input_message = "[TWT]<s>"
        token_ids = self.tokenizer.encode(input_message, add_special_tokens=False, return_tensors="pt")
        gen_args = {
                "max_length":1000,
                "min_length":15,
                "do_sample":True,
                "top_k":700,
                "top_p":0.50,
                "pad_token_id":self.tokenizer.pad_token_id,
                "bos_token_id":self.tokenizer.bos_token_id,
                "eos_token_id":self.tokenizer.eos_token_id,
                "bad_word_ids":[[self.tokenizer.unk_token_id]],
                "num_beams":random.randint(1,3),
                "early_stopping":random.choice([True,False]),
        }
        if random.random() > 0.5:
            del gen_args["num_beams"]
        if random.random() > 0.5:
            del gen_args["early_stopping"]

        with torch.no_grad():
            # print("generating")
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                **gen_args,
            )

        tweet = self.tokenizer.decode(output_ids.tolist()[0][2:])
        tweet = tweet.replace("</s>", "")
        return tweet

    def generate_reply(self, input_message):
        input_message = self.mention_user_pattern.sub("", input_message)
        input_message += "[REP][ほぼじゅき]<s>"
        
        token_ids = self.tokenizer.encode(
            input_message, add_special_tokens=False, return_tensors="pt")
        input_message_count = len(token_ids[0])
        while(True):
            gen_args = {
                "max_length":1000,
                "min_length":15,
                "do_sample":True,
                "top_k":700,
                "top_p":0.95,
                "pad_token_id":self.tokenizer.pad_token_id,
                "bos_token_id":self.tokenizer.bos_token_id,
                "eos_token_id":self.tokenizer.eos_token_id,
                "bad_word_ids":[[self.tokenizer.unk_token_id]],
                "num_beams":random.randint(1,3),
                "early_stopping":random.choice([True,False]),
            }

            with torch.no_grad():
                # print("generating")
                output_ids = self.model.generate(
                    token_ids.to(self.model.device),
                    **gen_args,
                )
            reply = self.tokenizer.decode(output_ids.tolist()[0][input_message_count:])
            reply = self.reply_sub_words.sub("", reply)
            reply = reply.replace("[]", "")
            if len(reply) >= 1:
                break
        return reply

    async def on_ready(self):
        self.mention_pattern = re.compile(f'<@{self.user.id}>')
        print('Logged on as', self.user)
        target_channel = self.get_channel(self.target_channel_id)
        tweet = self.generate_tweet()
        await target_channel.send(tweet)
        self.myloop.start()

    async def on_message(self, message):
        # dm
        if message.author != self.user:
            if message.author.dm_channel is None:
                await message.author.create_dm()
            if message.author.dm_channel.id == message.channel.id:
                await self.on_dm_message(message)
                return
        
        # hoboJUKI channel
        if message.channel.id == self.target_channel_id:
            await self.on_hoboJUKI_channel_message(message)
            return

    async def on_dm_message(self,message):
        if message.channel.id not in self.last_message_queues: 
            self.last_message_queues[message.author.dm_channel.id] = deque(maxlen=self.message_queue_max_length)
        message_queue = self.last_message_queues[message.author.dm_channel.id]
        if message.channel.id not in self.last_message_times:
            self.last_message_times[message.channel.id] = time.time()
        if time.time() - self.last_message_times[message.channel.id] > 3600:
            message_queue = deque(maxlen=self.message_queue_max_length)
        self.last_message_times[message.channel.id] = time.time()

        
        if message.author == self.user:
            message_queue.append(f"[REP][ほぼじゅき]<s>{message.content}</s>")
            return
        else:
            async with message.channel.typing():
                sleep_time = random.uniform(*self.reply_sleep_time_range)
                start = time.time()
                author_nick_name = message.author.name
                message_queue.append(f"[{author_nick_name}]<s>{message.content}</s>")
                reply_message = None
                reply_message = self.generate_reply("".join(message_queue))
                generate_end = time.time()
                if self.dm_logger is not None:
                    self.dm_logger.send_message(message.content,author_nick_name)
                    self.dm_logger.send_message(reply_message,"ほぼじゅき")
                if reply_message is not None:
                    if len(reply_message) > 4:
                        await asyncio.sleep(sleep_time-(generate_end-start))
                    await message.channel.send(reply_message)

        

    async def on_hoboJUKI_channel_message(self,message):
        message_queue = self.last_message_queues[message.channel.id]
        if message.channel.id not in self.last_message_times:
            self.last_message_times[message.channel.id] = time.time()
        if time.time() - self.last_message_times[message.channel.id] > 3600:
            message_queue = deque(maxlen=self.message_queue_max_length)
        self.last_message_times[message.channel.id] = time.time()
        author_nick_name = message.author.nick
        if author_nick_name is None:
            author_nick_name = message.author.name
        message_content = message.content
        # don't respond to ourselves
        if message.author == self.user:
            message_queue.append(f"[REP][ほぼじゅき]<s>{message_content}</s>")
            return
        else:
            async with message.channel.typing():
                sleep_time = random.uniform(*self.reply_sleep_time_range)
                start = time.time()
                message_queue.append(f"[{author_nick_name}]<s>{message_content}</s>")
                reply_message = None
                reply_message = self.generate_reply("".join(message_queue))
                generate_end = time.time()
                if reply_message is not None:
                    if len(reply_message) > 4:
                        await asyncio.sleep(sleep_time-(generate_end-start))
                    await message.channel.send(reply_message)

    @tasks.loop(minutes=15)
    async def myloop(self):
        if time.time() - self.last_message_times[self.target_channel_id] > 3600:
            if self.next_tweet_date is None:
                self.next_tweet_date = time.time() + random.randint(0,3600*6)
            else:
                if self.next_tweet_date - time.time() <= 0:
                    if False: # dispable tweet
                        # send tweet
                        target_channel = self.get_channel(self.target_channel_id)
                        tweet = self.generate_tweet()
                        await target_channel.send(tweet)
                    self.next_tweet_date = time.time() + random.randint(0,3600*6)

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = HoboJUKI(intents=intents)
    client.dm_logger = dmwebhook.hoboJUKIdmLogger(hoboJUKI_IDs.webhook_url)
    client.run(hoboJUKI_IDs.token)
