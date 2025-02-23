import os

# We rely on the environment for OPENAI_API_KEY:
#   export OPENAI_API_KEY="sk-123YourKeyHere"
# Then run: python rag_chroma_demo.py

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document

def main():
    # Synthetic documents
    docs = [
        Document(page_content="Apples are red and often used in pies."),
        Document(page_content="Bananas are yellow and a favorite among monkeys.")
    ]
    
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()
    
    # Build a Chroma vector store (ephemeral)
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # Create a retriever that fetches up to 2 docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    
    # Ask a question
    question = "What color are apples?"
    print(f"Q: {question}")
    
    # Instead of `chain({"query": question})`, use `.invoke`
    result = qa_chain.invoke({"query": question})
    
    # Extract the actual answer from the dict
    answer = result["result"]
    print(f"A: {answer}\n")

if __name__ == "__main__":
    main()

