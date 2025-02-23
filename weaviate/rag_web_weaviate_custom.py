import os
from typing import List
import requests
from bs4 import BeautifulSoup

import weaviate  # v3 client
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Weaviate as WeaviateVectorStore

def crawl_url(url: str) -> Document:
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text_content = soup.get_text(separator="\n")
    return Document(page_content=text_content, metadata={"source": url})

def main():
    urls_to_crawl: List[str] = ["https://example.com"]
    docs = [crawl_url(u) for u in urls_to_crawl]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Weaviate v3 client
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    client = weaviate.Client(url=weaviate_url)

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)

    class_name = "WebDocs"
    vectorstore = WeaviateVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding_fn,
        client=client,           # <-- CHANGED: 'client=' not 'weaviate_client='
        index_name=class_name,
        by_text=False
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    question = "What does the homepage mention?"
    response = qa_chain.invoke({"query": question})
    answer = response["result"]
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()

