import chainlit as cl
from rag_service import RAGService

rag_service = RAGService()

@cl.on_chat_start
async def main():
    await cl.Message(content="Hello, welcome to Chainlit!").send()
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

@cl.on_message
async def message_handler(message: cl.Message):
    try:
        message_history = cl.user_session.get("message_history")
        message_history.append({"role": "user", "content": message.content})

        stream = await rag_service.query_document_with_openai(message.content, message_history)

        msg = cl.Message(content="")
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()
    except Exception as e:
        print(f"Error in message handler: {str(e)}")
        await cl.Message(content="Sorry, an error occurred while processing your request").send()