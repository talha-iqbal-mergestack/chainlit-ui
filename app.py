import chainlit as cl
import aiohttp

@cl.on_chat_start
async def main():
    await cl.Message(content="Hello, welcome to Chainlit!").send()

@cl.on_message
async def message_handler(message: cl.Message):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:5000/api/v1/documents/query",
            json={"query": message.content}
        ) as response:
            if response.status == 200:
                data = await response.json()
                await cl.Message(content=data["answer"]).send()
            else:
                await cl.Message(content="Sorry, an error occurred while processing your request").send()