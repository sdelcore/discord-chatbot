import discord
import model_wrapper

import os
from dotenv import load_dotenv, find_dotenv

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

async def send_message(message_ctx, msg):
    user = message_ctx.author.mention
    await message_ctx.channel.send('{} {}'.format(user, msg))

@client.event
async def on_message(message):
    client_dm = message.content.split(' ')[0]
    
    if (message.author == client.user) or (str(client.user.id) not in client_dm):
        return
    
    input_sentence = message.content[len(client_dm)+1:]

    if '!personality' in input_sentence:
        await send_message(message, model_wrapper.get_personality())
    else:
        resp = model_wrapper.chat(input_sentence)
        await send_message(message, resp)

def main():
    global client
    load_dotenv('.env')
    DISCORD_KEY = os.environ.get("DISCORD_KEY")
    
    # load model
    model_wrapper.init()
    
    # START DISCORD API
    
    client.run(DISCORD_KEY)

if __name__ == '__main__':
    main()
