import chainlit as cl
from rag_service import RAGService

rag_service = RAGService()

@cl.on_chat_start
async def main():
    await cl.Message(content="Hello, welcome to Chainlit!").send()

@cl.on_message
async def message_handler(message: cl.Message):
    try:
        response = await rag_service.query_document_with_openai(message.content)
        await cl.Message(content=response["answer"]).send()
    except Exception as e:
        await cl.Message(content="Sorry, an error occurred while processing your request").send()