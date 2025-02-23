import os
from typing import Optional, List, Dict

# 1) Pinecone (new library)
from pinecone import Pinecone, ServerlessSpec

# 2) langchain_community imports for AWS S3
from langchain_community.document_loaders import S3DirectoryLoader

# 3) langchain_openai imports for LLM and embeddings (replaces older classes)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI

# 4) langchain (core) imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


def main():
    # ----------------------------------------------------------------------
    # 1) LOAD DOCUMENT(S) FROM S3
    # ----------------------------------------------------------------------
    bucket_name = "my-demo-bucket-2025-unique"  # your bucket
    prefix = "fruit_data.txt"                   # or "" if you want all root objects

    loader = S3DirectoryLoader(
        bucket=bucket_name,
        prefix=prefix,
        region_name="us-east-1"  # adjust if your bucket is in another region
    )
    print("Loading documents from S3...")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from s3://{bucket_name}/{prefix}.\n")

    # ----------------------------------------------------------------------
    # 2) SPLIT THE DOCUMENT INTO CHUNKS
    # ----------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunk(s).\n")

    # ----------------------------------------------------------------------
    # 3) INITIALIZE PINECONE
    # ----------------------------------------------------------------------
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "YOUR-PINECONE-KEY")
    pinecone_env = os.environ.get("PINECONE_ENV", "us-east-1")

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

    index_name = "quickstart"
    existing_indices = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indices:
        # Create an index matching your embedding dimension (e.g., 1536 for text-embedding-ada-002)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created Pinecone index '{index_name}' (dimension=1536).\n")

    # Retrieve an index client (NOTE: use pc.Index(...) not pc.index(...))
    index_client = pc.Index(index_name)
    print(f"Using Pinecone index: {index_name}\n")

    # ----------------------------------------------------------------------
    # 4) SET UP EMBEDDINGS
    # ----------------------------------------------------------------------
    # Requires `pip install langchain-openai`
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # ----------------------------------------------------------------------
    # 5) UPSERT DOCUMENTS INTO PINECONE
    # ----------------------------------------------------------------------
    def upsert_documents(document_list, embedding_function, pinecone_index):
        """Converts each chunk to an embedding and upserts into Pinecone."""
        vectors_to_upsert = []
        for i, doc in enumerate(document_list):
            text = doc.page_content
            metadata = dict(doc.metadata)

            emb = embedding_function.embed_query(text)
            vector_id = f"doc_{i}"

            # store the chunk text in metadata for retrieval
            metadata["page_content"] = text
            vectors_to_upsert.append((vector_id, emb, metadata))

        pinecone_index.upsert(vectors=vectors_to_upsert)

    print("Upserting doc chunks to Pinecone...")
    upsert_documents(split_docs, embedding_fn, index_client)
    print("Upsert complete.\n")

    # ----------------------------------------------------------------------
    # 6) DEFINE A CUSTOM PINECONE VECTORSTORE
    # ----------------------------------------------------------------------
    class PineconeVectorStore(VectorStore):
        """Minimal custom Pinecone VectorStore that implements required methods."""

        def __init__(self, pinecone_index, embedding_function):
            self.index = pinecone_index
            self.embedding_fn = embedding_function

        def similarity_search(self, query: str, k: int = 4):
            """Return top-k doc chunks from Pinecone."""
            query_embedding = self.embedding_fn.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            docs_out = []
            for match in results.matches:
                md = match.metadata or {}
                text_content = md.get("page_content", "")
                docs_out.append(Document(page_content=text_content, metadata=md))
            return docs_out

        @classmethod
        def from_texts(
            cls,
            texts: Optional[List[str]],
            embedding: OpenAIEmbeddings,
            metadatas: Optional[List[Dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs
        ):
            raise NotImplementedError("Use upsert_documents manually.")

        @classmethod
        def from_documents(cls, documents, embedding, **kwargs):
            raise NotImplementedError("Use upsert_documents manually.")

        def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs
        ) -> List[str]:
            raise NotImplementedError("Use upsert_documents manually.")

    # Create the store and a retriever
    vectorstore = PineconeVectorStore(index_client, embedding_fn)
    retriever = vectorstore.as_retriever()

    # ----------------------------------------------------------------------
    # 7) BUILD RETRIEVALQA WITH OPENAI LLM
    # ----------------------------------------------------------------------
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # ----------------------------------------------------------------------
    # 8) ASK A TEST QUESTION
    # ----------------------------------------------------------------------
    question = "What fruit information is in fruit_data.txt?"

    # Instead of `chain.__call__`, we can do .invoke() or pass a dict:
    response = qa_chain.invoke({"query": question})
    # or: response = qa_chain({"query": question})

    # Access the chain's result
    answer = response["result"]
    print(f"Answer:\n{answer}\n")


if __name__ == "__main__":
    main()

