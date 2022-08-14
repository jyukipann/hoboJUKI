import discord
import hoboJUKI_IDs
import re
from discord.ext import tasks


class HoboJUKI(discord.Client):
    def __init__(self, *, intents, **options) -> None:
        super().__init__(intents=intents, **options)
        self.target_channel_id = 790812986599931967

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

    @tasks.loop()
    async def loop():
        print("loop")




if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = HoboJUKI(intents=intents)
    client.run(hoboJUKI_IDs.token)
