from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ArxivRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from arxiv_chat.arxivpdf import ArxivPDF
import argparse
from pathlib import Path
import chromadb
import logging
from langchain.schema import BaseRetriever, Document

CHROMA_DB_DIR = "./chroma_db"


def format_front_matter(abstract_metadata):
    out = ""
    for k, v in abstract_metadata.items():
        out += f"{k}: {v}\n\n"
    return Document(page_content=out, metadata=abstract_metadata)


def main(fname: str, force_overwrite:bool=True, live_input:bool=True):
    
    fname = "2302.00923"
    # retriever = ArxivRetriever(load_max_docs=200)
    #import pdb; pdb.set_trace()
    #front_matter = retriever.load(query=fname)
    #docs = retriever.get_relevant_documents(fname)

    pdf = ArxivPDF()
    front_matter, body = pdf.load(query=fname, parse_pdf=True)
    #docs = body
    header = format_front_matter(front_matter[0].metadata)
    docs = [header] + body

    # Define our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    # Build vectorstore and keep the metadata
    collection_name = fname
    #persist_dir = vector_db_dir / collection_name
    embeddings_obj = OpenAIEmbeddings()
    embedding_function = embeddings_obj.embed_documents
    persistent_client = chromadb.PersistentClient(CHROMA_DB_DIR)
    collections = set([col.name for col in persistent_client.list_collections()])
    print(f"Existing collections: {collections}")

    if not force_overwrite and collection_name in collections:
        print(f"Loading {collection_name} from disk")
        # load from disk
        collection = persistent_client.get_collection(collection_name)
        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings_obj,
        )
        # vectorstore = Chroma(persist_directory=str(persist_dir), embedding_function=embedding_function)
    else:
        if force_overwrite:
            print(f"Creating {collection_name} and saving to disk")
            persistent_client.delete_collection(collection_name)
        # create and save to disk
        collection = persistent_client.create_collection(
            collection_name,
            embedding_function=embedding_function,
        )

        collection.add(
            ids=[str(i) for i in range(len(all_splits))],
            documents=[doc.page_content for doc in all_splits],
            metadatas=[doc.metadata for doc in all_splits],
        )

        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings_obj,
        )

    # Define our metadata
    metadata_field_info = [
        AttributeInfo(
            name="Section",
            description="Part of the document that the text comes from",
            type="string or list[string]",
        ),
    ]
    document_content_description = "Major sections of the document"

    # Define self query retriver
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
    # out = retriever.get_relevant_documents("Summarize the introduction section of the document")
    # print(out)
    llm = ChatOpenAI(temperature=0, verbose=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever,
        verbose=True,
        return_source_documents=False,
        memory=ConversationBufferMemory()
    )
    #out = qa_chain("Summarize the motivation section of this document")
    if live_input:
        while True:
            question = input("Enter a question (q to quit): ")
            if question.strip() == "q":
                break
            answer = qa_chain(question)
            print(f"query: {answer['query']}")
            print(f"result: {answer['result']}")
    else:
        out = qa_chain("Give a concise summary of this document")
        print(out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run doc QA on the given Arxiv PDF")
    parser.add_argument("fname", type=str, help="Path to the PDF file")
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Force overwrite of existing Chroma DB")
    parser.add_argument("--live_input", action="store_true", help="Live input mode")
    #parser.add_argument("--vector_db", type=Path, help="Path to the Chroma DB", default="./chroma_db")
    args = parser.parse_args()
    main(args.fname, args.force_overwrite, args.live_input)
