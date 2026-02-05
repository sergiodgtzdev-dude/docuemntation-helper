from typing import Any, Dict
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

model = init_chat_model(model="gpt-5.2", model_provider="openai")


# tool will return two values
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get("source", "Unknown")} \n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return both docs and serialized content
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation
    :param query: The user's question
    :return:
        Dictionary containing:
            - answer: The generated answer
            - context: The list of retrieved documents
    """
