from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever, Document
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from arxiv_chat.arxivpdf import ArxivPDF
import argparse
import chromadb
from typing import List
import gradio as gr

CHROMA_DB_DIR = "./chroma_db"


def format_front_matter(abstract_metadata: dict) -> Document:
    out = ""
    for k, v in abstract_metadata.items():
        out += f"{k}: {v}\n\n"
    return Document(page_content=out, metadata=abstract_metadata)


def create_vector_store(
        docs: List[Document], collection_name: str, force_overwrite:bool=True
    )-> Chroma:
    """
    Create a vectorstore from a list of documents
    """
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
            ids=[str(i) for i in range(len(docs))],
            documents=[doc.page_content for doc in docs],
            # metadatas=[doc.metadata for doc in all_splits],
        )

        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings_obj,
        )
    return vectorstore

def create_docQA_chain(fname: str, force_overwrite:bool=True):
    pdf = ArxivPDF()
    front_matter, body, doc_file_name = pdf.load(query=fname, parse_pdf=True, split_sections=False, keep_pdf=True)
    header = format_front_matter(front_matter[0].metadata)
    docs = [header] + body

    # Define our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    vectorstore = create_vector_store(all_splits, fname, force_overwrite)
    # Define our metadata
    # metadata_field_info = [
    #     AttributeInfo(
    #         name="Section",
    #         description="Part of the document that the text comes from",
    #         type="string or list[string]",
    #     ),
    # ]
    # document_content_description = "Major sections of the document"

    # Define self query retriver this uses the LLM to generate vector store queries
    llm = OpenAI(temperature=0)
    # retriever = SelfQueryRetriever.from_llm(
    #     llm, vectorstore,
    #     #document_content_description,
    #     #metadata_field_info,
    #     verbose=True)

    # out = retriever.get_relevant_documents("Summarize the introduction section
    # of the document")
    llm = OpenAI(temperature=0, verbose=True)
    qa_chain =  RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        verbose=True,
        return_source_documents=False,
    )
    return qa_chain, doc_file_name, front_matter[0].metadata

fnames = [
        "2302.00923",
        "2211.11559", 
]

def main(force_overwrite:bool=True, live_input:bool=True):
    
    tools = []
    for fname in fnames:
        qa_chain, file_name, metadata = create_docQA_chain(fname, force_overwrite)
        tool = Tool(
            name=fname,
            func=qa_chain,
            description=f"""
            useful for when you need to answer questions about the paper titled {metadata['Title']}. Input should be a fully formed question.
            """ # by {metadata['Authors']}
            # published on {metadata['Published']}.
            # """,u
            # agent_type=AgentType.RETRIEVAL_QA,
        )
        print(tool.description)
        tools.append(tool)

    llm = ChatOpenAI(temperature=0, verbose=True)
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    if live_input:
        history = []
        while True:
            question = input("Enter a question (q to quit): ")
            if question.strip() == "q":
                break
            answer = agent.run(input=question)
            print(answer)
        # def llm_response(question, history=None):
        #     answer = agent.run(question)
        #     return answer['result'] 
        # gr.ChatInterface(llm_response).launch()
    else:
        questions = [
            # "What is multimodal chain-of-thought reasoning?",
            # "What are the main contributions of the Multimodal Chain-of-Though Reasoning paper?",
            # "What are the main contributions of the Visual QA paper?",
            # "What methods are used in the Visual QA paper?",
            # "How could visual QA be improved?",
            "Give me a concise summary of the visual QA paper"
            # "Who wrote this paper?",
            # "What is the title of this paper?",
            # "What are the main contributions?",
            # "What methods are used?",
            # "What are the results?",
            # "Give you give a concise summary of this document?"
        ]
        for question in questions:
            print(f"query: {question}")
            out = agent.run(question)
            print(f"result: {out}")
            print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run doc QA on the given Arxiv PDF",
        usage="python run.py 2302.0092"
    )
    # parser.add_argument(
    #     "--fname",
    #     type=str, help="The Arxiv ID of the paper to run QA on",
    #     default="2302.0092",
    #     # Restrict to the following two papers for now until parsing is more robust
    #     choices = [
    #         "2302.00923",
    #         "2211.11559", 
    #         #"2307.09288", 
    #     ]
    # )
    #parser.add_argument("fname", type=str, help="Path to the PDF file")
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Force overwrite of existing Chroma DB")
    parser.add_argument("--live_input", action="store_true", help="Live input mode")
    args = parser.parse_args()
    main(args.force_overwrite, args.live_input)
