from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
import chainlit as cl
import os

class RAGService:
    def __init__(self):
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = 'document-store'
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    async def query_document_with_langchain(self, query: str) -> dict:
        # Initialize retriever directly from the existing index
        index = self.pinecone_client.Index(self.index_name)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",
            namespace="fitz_large"
        )

        retriever = vectorstore.as_retriever()
        
        # Define assistant prompt template
        prompt_template = PromptTemplate(
            template="""
            Answer the question based only on the following context:
            {context}
            Answer the question based on the above context: {question}.
            Provide a detailed answer.
            Don’t justify your answers.
            Don’t give information not mentioned in the CONTEXT INFORMATION.
            Do not say "according to the context" or "mentioned in the context" or similar.
            If the answer includes a table, preserve its structure using markdown table formatting.
            """,
            input_variables=["context", "question"]
        )

        # Create QA chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4.1",
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response
        response = qa_chain({"query": query})
        
        return {
            "answer": response["result"],
            "sources": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content[:300]
                }
                for doc in response["source_documents"]
            ]
        }

    async def query_document_with_openai(self, query: str, message_history: list = []) -> dict:
        # Initialize OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get embeddings for the query using OpenAI
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding

        # Query Pinecone
        index = self.pinecone_client.Index(self.index_name)
        query_response = index.query(
            vector=query_embedding,
            top_k=4,
            namespace="fitz",
            include_metadata=True
        )

        # Prepare context from matched documents
        contexts = []
        sources = []
        for match in query_response.matches:
            if match.metadata.get("text"):
                contexts.append(match.metadata["text"])
                sources.append({
                    "metadata": match.metadata,
                    "content": match.metadata["text"][:300],
                    "score": match.score  # Adding the match score
                })

        # Prepare prompt with context
        context_str = "\n\n".join(contexts)
        prompt=f"""
        Answer the question based only on the following context:
        {context_str}
        Answer the question based on the above context: {query}.
        Provide a detailed answer.
        Don’t justify your answers.
        Don’t give information not mentioned in the CONTEXT INFORMATION.
        Do not say "according to the context" or "mentioned in the context" or similar.
        If the answer includes a table, preserve its structure using markdown table formatting.
        """

        # Modify the chat completion call to include message history
        messages = message_history
        messages.append({"role": "user", "content": prompt})

        # Get completion from OpenAI
        stream = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True
        )

        # Print scores with formatted output
        print("\nSimilarity Scores:")
        print("-----------------")
        for source in sources:
            print(f"Score: {source['score']:.4f}")

        return stream