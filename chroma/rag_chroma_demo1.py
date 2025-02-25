import os

# We rely on the environment for OPENAI_API_KEY:
#   export OPENAI_API_KEY="sk-123YourKeyHere"
# Then run: python rag_chroma_demo.py

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document

def main():
    # 1) Read from local "fruit.txt"
    with open("fruit.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    # 2) Create a Document object from file contents
    docs = [Document(page_content=text_data)]
    
    # 3) Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()
    
    # 4) Build a Chroma vector store (ephemeral)
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # 5) Create a retriever that fetches up to 2 docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # 6) Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    
    # 7) Ask a question
    question = "What color are apples?"
    print(f"Q: {question}")
    
    # 8) Use `.invoke` to query the chain
    result = qa_chain.invoke({"query": question})
    
    # 9) Extract the answer from the result
    answer = result["result"]
    print(f"A: {answer}\n")

if __name__ == "__main__":
    main()

