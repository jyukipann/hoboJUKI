import discord
import hoboJUKI_IDs
import re
from collections import deque
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM


class HoboJUKI(discord.Client):
    def __init__(self, *, intents, **options) -> None:
        super().__init__(intents=intents, **options)

    async def on_ready(self):
        self.mention_pattern = re.compile(f'<@{self.user.id}>')
        print('Logged on as', self.user)

    async def on_message(self, message):
        if message.author == self.user:
            return
        if message.author.dm_channel is None:
            await message.author.create_dm()
        if message.author.dm_channel.id == message.channel.id:
            print("here is dm")



if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = HoboJUKI(intents=intents)
    client.run(hoboJUKI_IDs.token)
