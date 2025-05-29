from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import numpy as np
import os

class RAGService:
    def __init__(self):
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = 'document-store'
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    async def query_document(self, query: str) -> dict:
        # Initialize retriever directly from the existing index
        index = self.pinecone_client.Index(self.index_name)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",
            namespace="fitz_large"
        )

        retriever = vectorstore.as_retriever()
        
        # Define HR assistant prompt template
        prompt_template = PromptTemplate(
            template="""
            You are an HR assistant. Answer the question using the provided context.
            If the answer includes a table, preserve its structure using markdown table formatting.

            Context:
            {context}

            Question: {question}
            Answer:
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

    async def query_document_with_openai(self, query: str) -> dict:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get embeddings for the query using OpenAI
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding

        # Query Pinecone
        index = self.pinecone_client.Index(self.index_name)
        query_response = index.query(
            vector=query_embedding,
            top_k=4,
            namespace="pdfplumber",
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
        prompt = f"""You are an HR assistant. Answer the question using the provided context.
        If the answer includes a table, preserve its structure using markdown table formatting.

        Context:
        {context_str}

        Question: {query}
        Answer:"""

        # Get completion from OpenAI
        chat_completion = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an HR assistant. Provide accurate answers based on the given context."},
                {"role": "user", "content": prompt}
            ]
        )

        # Print scores with formatted output
        print("\nSimilarity Scores:")
        print("-----------------")
        for source in sources:
            print(f"Score: {source['score']:.4f}")

        return {
            "answer": chat_completion.choices[0].message.content,
            "sources": sources
        }